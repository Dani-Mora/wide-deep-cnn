from protodata.data_ops import get_interval_mask

import abc
from abc import ABCMeta
import numpy as np
import tensorflow as tf


""" Helper class to implement network metrics in order to evaluate
the model performance """


class Metric(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.op = None

    def compute(self, logits, targets, add_summary=True):
        """ Returns the value of the defined metric """
        # Whether metric needs to be tracked or not
        track = False if self.never_track() else add_summary

        # Transformation before metric computation
        predicted, targets = self._preprocess(logits, targets)

        # Check compatibility if needed
        predicted.get_shape().assert_is_compatible_with(targets.get_shape())

        with tf.variable_scope(self.get_label()):
            return self._compute(predicted, targets, track)

    @abc.abstractmethod
    def _compute(self, predicted, targets, track):
        """ Returns the value of the defined metric """

    @abc.abstractmethod
    def _preprocess(self, logits, targets):
        """ Transforms logits and targets to be compatible with each other
        in the metric context """

    @abc.abstractmethod
    def get_label(self):
        """ Returns the label (name) of the metric """

    @abc.abstractmethod
    def reduce_mean(self, values):
        """ Returns the mean value of the input metric values """

    def compute_op(self, logits, targets, add_summary=True):
        """ Computes and saves the Tensorflow metric operation """
        self.op = self.compute(logits, targets, add_summary)

    def never_track(self):
        """ Function to be overriden for those metrics we never want to
        store """
        return False

    def get_op(self):
        """ Returns the Tensorflow metric operation """
        if self.op is None:
            raise ValueError("%s: Operation undefined. Call compute_op before"
                             % self.get_label())
        return self.op

    @abc.abstractmethod
    def write_summary(self, value, writer, step):
        """ Operation to explicitely track a summary """


class ClassificationMetric(Metric):

    """ Classification metrics expect to receive two vectors of the size
    of the batch: one for the predicted labels and one for the groundtruth
    """

    __metaclass__ = ABCMeta

    def _preprocess(self, logits, targets):
        # Get most probable class of the output
        logits = tf.arg_max(logits, dimension=1)

        # If one-hot provided, transform into class
        if targets.get_shape().ndims > 2:
            targets = tf.arg_max(targets, dimension=1)

        # Erase singletion dimension if exists
        if targets.get_shape().ndims > 1:
            targets = tf.squeeze(targets, axis=1)

        return logits, targets


class RegressionMetric(Metric):

    """ Regression metrics expect to receive two vectors of the size of
    the batch and having a depth given by the specific problem
    """

    __metaclass__ = ABCMeta

    def _preprocess(self, logits, targets):
        return logits, targets


class SingleMetric(Metric):

    """ Metric based on a single value """

    __metaclass__ = ABCMeta

    def _compute(self, predicted, targets, track):
        # If targets are encoded in one-hot, transform into classes
        # Compute specific loss
        metric_value = self._compute_specific(predicted, targets)

        # Add Tensorboard summary, if requested
        if track:
            tf.summary.scalar(name=self.get_label(), tensor=metric_value)
        return metric_value

    def reduce_mean(self, values):
        return np.mean(values)

    @abc.abstractmethod
    def _compute_specific(self, logits, targets):
        """ Returns the value of the defined metric """

    def write_summary(self, value, writer, step):
        summ_tag = '/'.join([self.get_label(), self.get_label()])
        summ = tf.Summary(
            value=[tf.Summary.Value(tag=summ_tag, simple_value=float(value))]
        )
        writer.add_summary(summ, step)


class IntervalMetric(Metric):

    """ Computes a metric on separated intervals of the groundtruth data """

    def __init__(self, cuts):
        super(Metric, self).__init__()
        self.cuts = cuts

    def interval_name(self, name):
        return '_'.join([self.get_label(), name])

    def define_names(self):
        names = []
        for i in range(len(self.cuts)):

            if i == 0:
                name = 'minus_' + str(self.cuts[i])
            else:
                name = "between_" + str(self.cuts[i-1]) \
                       + "_and_" + str(self.cuts[i])
            names.append(name)

            # Add extra name for last bin
            if i == len(self.cuts)-1:
                names.append('above_' + str(self.cuts[i]))

        return names

    def _compute(self, predicted, targets, track):

        batch_size = tf.shape(targets)[0]

        def interval_value(mask, name):
            """ Returns value of the interval defined by the mask and tracks
            it, if needed """
            # Compute masked metric
            mask = tf.reshape(mask, [batch_size])
            value = self._masked_metric(targets, predicted, mask)
            # Add summary if requested
            if track:
                tf.summary.scalar(self.interval_name(name), value)
            return value

        # Lis of metric values
        intervals = []
        names = self.define_names()
        for i in range(len(self.cuts)):

            if i == 0:
                # Exclusively lower than cuts[0]
                mask = tf.less(targets, self.cuts[i])

            else:

                # Case cuts[i-1] <= x < cuts[i]
                mask = get_interval_mask(low=self.cuts[i - 1],
                                         high=self.cuts[i],
                                         values=targets)

            # Add current bin
            intervals.append(interval_value(mask=mask, name=names[i]))

            # Extra case: exclusively greater or equal to cuts[-1]
            if i == len(self.cuts) - 1:
                mask_extra = tf.greater_equal(targets, self.cuts[i])
                intervals.append(
                    interval_value(mask=mask_extra, name=names[-1])
                )

        return tf.stack(intervals)

    @abc.abstractmethod
    def get_metric_func(self):
        """ Returns the metric function for the subclass """

    def _masked_metric(self, groundt, predicted, mask):
        """ Computs the metric of the input arrays considering only the positions
        in the boolean mask """
        groundt_masked = tf.boolean_mask(groundt, mask)
        predicted_masked = tf.boolean_mask(predicted, mask)
        func = self.get_metric_func()
        return func(groundt_masked, predicted_masked)

    def reduce_mean(self, values):
        return np.asarray(np.nanmean(np.asmatrix(values), axis=0))

    def write_summary(self, value, writer, step):
        # Check lengths are consistent
        names = self.define_names()
        value = np.squeeze(value)
        if len(value) != len(names):
            raise RuntimeError('Provided list of values is not consistent '
                               + ' with bins defined.')
        # Store each summary independently
        for i, n in enumerate(names):
            summ_name = self.interval_name(names[i])
            summ_tag = '/'.join([self.get_label(), summ_name])
            summ = tf.Summary(
                value=[tf.Summary.Value(tag=summ_tag,
                                        simple_value=float(value[i]))]
            )
            writer.add_summary(summ, step)


""" Classification metrics """


class Accuracy(SingleMetric, ClassificationMetric):

    """ Given two label predictions, returns the agreement accuracy """

    def _compute_specific(self, predicted, targets):
        same = tf.equal(predicted, tf.cast(targets, tf.int64))
        return tf.reduce_mean(tf.cast(same, tf.float32))

    def get_label(self):
        return "accuracy"


class AccuracyMode(SingleMetric, ClassificationMetric):

    """ Tracks the accuracy resulted when predicting the most popular
    class within the groundtruth batch """

    def __init__(self, num_classes):
        super(AccuracyMode, self).__init__()
        self._num_classes = num_classes

    def _compute_specific(self, predicted, targets):
        eq = tf.equal(x=get_mode(targets, self._num_classes),
                      y=tf.cast(targets, tf.int64))
        return tf.reduce_mean(tf.cast(eq, tf.float32))

    def get_label(self):
        return "accuracy_mode"


class AccuracyRandom(SingleMetric, ClassificationMetric):

    """ Tracks the accuracy resulted when predicting random valid classes """

    def __init__(self, num_classes):
        super(AccuracyRandom, self).__init__()
        self._num_classes = num_classes

    def _compute_specific(self, predicted, targets):
        # Generate random predictions with equal probabilities
        random = tf.multinomial(logits=tf.log([[10.] * self._num_classes]),
                                num_samples=tf.shape(targets)[0])
        # Output prediction as ratio of matches
        eq = tf.equal(x=random, y=tf.cast(targets, tf.int64))
        return tf.reduce_mean(tf.cast(eq, tf.float32))

    def get_label(self):
        return "accuracy_random"


class ConfusionMatrix(SingleMetric, ClassificationMetric):

    """ Generates a confusion matrix given thge predicted classes and the
    targets """

    def __init__(self, num_classes):
        super(ConfusionMatrix, self).__init__()
        self._num_classes = num_classes

    def _compute_specific(self, predicted, targets):
        targets = tf.cast(targets, tf.int64)
        return tf.confusion_matrix(labels=targets,
                                   predictions=predicted,
                                   num_classes=self._num_classes)

    def get_label(self):
        return "conf_matrix"

    def never_track(self):
        return True

    # Override mean to adjust it to matrices
    def reduce_mean(self, values):
        return np.around(np.add.reduce(values) / float(len(values)),
                         decimals=1)


""" Regression metrics """


class AbsError(SingleMetric, RegressionMetric):

    """ Returns the absolute error """

    def _compute_specific(self, predicted, targets):
        return absolute_error(targets, predicted)

    def get_label(self):
        return "abs_error"


class Std(SingleMetric, RegressionMetric):

    """ Returns the standard deviation of the output """

    def _compute_specific(self, predicted, targets):
        return tf.squeeze(tf.sqrt(tf.nn.moments(predicted, axes=[0])[1]),
                          squeeze_dims=[0])

    def get_label(self):
        return "out_std"


class Mean(SingleMetric, RegressionMetric):

    """ Returns the mean of the output """

    def _compute_specific(self, predicted, targets):
        return tf.reduce_mean(predicted)

    def get_label(self):
        return "out_mean"


class RMSEInterval(IntervalMetric, RegressionMetric):

    """ Returns the RMSE score on different intervals of the groundtruth """

    def __init__(self, cuts):
        super(RMSEInterval, self).__init__(cuts)

    def get_metric_func(self):
        return rmse

    def get_label(self):
        return "rmse_interval"


class AbsErrorInterval(IntervalMetric, RegressionMetric):

    """ Returns the absolute error on different intervals of the
    groundtruth """

    def __init__(self, cuts):
        super(AbsErrorInterval, self).__init__(cuts)

    def get_metric_func(self):
        return absolute_error

    def get_label(self):
        return "abs_err_interval"


class MeanPredError(SingleMetric, RegressionMetric):

    """ Returns the error we would make if we predicted the mean """

    def _compute_specific(self, predicted, targets):
        mean_tensor = tf.ones(shape=tf.shape(targets)) \
            * tf.reduce_mean(targets)
        return absolute_error(mean_tensor, targets)

    def get_label(self):
        return "error_mean_pred"


class MedianPredError(SingleMetric, RegressionMetric):

    """ Returns the error we would make if we predicted the median """

    def _compute_specific(self, predicted, targets):
        total = tf.shape(targets)[0]
        # Sort targets and take value in the middle
        targets = tf.squeeze(targets, squeeze_dims=[targets.get_shape()[-1]])
        sorted_targets, _ = tf.nn.top_k(targets, k=total)
        median = tf.gather(sorted_targets, 
                           tf.cast(tf.floor([total/2]), tf.int32))
        # Build tensor with medians and return abs error
        median_tensor = tf.ones(shape=tf.shape(targets)) * median
        return absolute_error(median_tensor, targets)

    def get_label(self):
        return "error_median_pred"


class ErrorPredicted(SingleMetric, RegressionMetric):

    """ Returns the error that would incur if we always predicted the
    given value """

    def __init__(self, value):
        super(ErrorPredicted, self).__init__()
        self.value = value

    def _compute_specific(self, predicted, targets):
        return absolute_error(targets, tf.constant(self.value))

    def get_label(self):
        return "predicted_" + str(self.value)


def rmse(groundt, predicted):
    """ Computes the RMSE of the input arrays """
    return tf.sqrt(
        tf.reduce_mean(tf.square(tf.cast(groundt, tf.float32) -
                       tf.cast(predicted, tf.float32)))
    )


def absolute_error(groundt, predicted):
    """ Returns the mean absolute different between the predicted data
    and the groundtruth """
    return tf.reduce_mean(tf.abs(groundt - predicted))


def get_mode(x, num_labels):
    """ Returns the mode in the input vector of labels.
    WARNING: assumes the values of the vector are in interval [0, num_labels).
    """
    if x.get_shape().ndims > 1:
        raise RuntimeError('Input must be a one dimensional tensor')
    # Sum columns of one hot representation
    one_hot = tf.one_hot(x, depth=num_labels, on_value=1, off_value=0)
    summed = tf.reduce_sum(one_hot, axis=0)
    # Return label with most support
    return tf.argmax(summed, dimension=0)
