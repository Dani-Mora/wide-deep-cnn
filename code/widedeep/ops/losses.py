"""
Helper class for implemented loss functions
"""

from protodata.data_ops import copy_columns
from widedeep.utils import get_logger, has_method

import abc
import tensorflow as tf


logger = get_logger('data')


# Enum of popular optimizers
class Optimizers:
    SGD = tf.train.GradientDescentOptimizer
    Adagrad = tf.train.AdagradOptimizer
    Adam = tf.train.AdamOptimizer
    Momentum = tf.train.MomentumOptimizer
    Ftrl = tf.train.FtrlOptimizer
    RMS = tf.train.RMSPropOptimizer


class LossFn(object):

    __metaclass__ = abc.ABCMeta

    def compute_loss(self, logits, targets):
        """ Returns the loss given the output of the model and
        the groundtruth """
        with tf.variable_scope('loss'):
            # Check they are compatible to avoid unexpected behaviors
            processed_targets = self.preprocess(logits, targets)
            if self.check_compatibility():
                logits.get_shape().assert_is_compatible_with(
                    processed_targets.get_shape()
                )
            # Get mean
            return tf.reduce_mean(self.loss(logits, processed_targets)) * \
                self.get_weight_term()

    def check_compatibility(self):
        return True

    @abc.abstractmethod
    def loss(self, logits, targets):
        """Returns the specific loss between output and groundtruth """

    @abc.abstractmethod
    def preprocess(self, logits, targets):
        """ Preprocess logits so they match the shape of the labels """

    def get_weight_term(self):
        return 1.0


class ClassificationLoss(LossFn):
    """ Abstract class for losses in classification tasks """

    __metaclass__ = abc.ABCMeta

    def __init__(self, smooth_label=None, smooth_neigh=None):
        """ Initializes the classification loss, which uses the one-hot
        encoding of the targets. Smoothing options are available. If none
        selected, classic one-hot encoding is performed.

        Args:
            smooth_label: If not None, this is the score assigned to the true
                label in the one hot encoding. The remaining probability is
                equally distributed among the other classes, ensuring all
                classes have non-zero probability.

            Example for label 3 in a 4-class problem and smooth_label of 0.7:

                - One hot vector: [0.0, 0.0, 1.0, 0.0]
                - Smoothed: [0.1, 0.1, 0.7, 0.1]

            smooth_neigh: If not None, this is the score assigned to the
                label in the one hot encoding. The remaining probability is
                equally distributed among the neighbouring classes in the one
                hot vector. This is useful when class represent some order.

            Example for label 3 in a 4-class problem and smooth_label of 0.7:

                - One hot vector: [0.0, 0.0, 1.0, 0.0]
                - Smoothed: [0.0, 0.15, 0.7, 0.15]
        """
        if smooth_label is not None and smooth_neigh is not None:
            raise ValueError('Cannot provide a valid value for both' +
                             ' smoothing strategies. Select either of them')
        if smooth_label is not None \
                and (smooth_label >= 1.0 or smooth_label <= 0):
            raise ValueError('Value must be in interval (0, 1)')
        if smooth_neigh is not None \
                and (smooth_neigh >= 1.0 or smooth_neigh <= 0):
            raise ValueError('Value must be in interval (0, 1)')
        self.smooth_label = smooth_label
        self.smooth_neigh = smooth_neigh

    def preprocess(self, logits, targets):
        """ Converts one dimensional target into one-hot enconding """
        # Squeeze vector in case second dimension is singleton

        num_classes = logits.get_shape()[-1]

        # Squeeze singleton dimension if exists
        if targets.get_shape().ndims == 2 and targets.get_shape()[-1] == 1:
            targets = tf.squeeze(targets, axis=[1])

        # If more than 1 dimension, one hot is already computed
        if targets.get_shape().ndims == 1:
            # Convert into One-hot encoding according to selected strategy
            if self.smooth_label is not None:
                logger.info("Smooth labeling using soft value {}"
                            .format(self.smooth_label))
                targets = smooth_labels(targets,
                                        self.smooth_label,
                                        num_classes)
            elif self.smooth_neigh is not None:
                logger.info("Smooth labeling with soft value {} for neighbors"
                            .format(self.smooth_neigh))
                targets = smooth_neighbouring(targets,
                                              self.smooth_neigh,
                                              num_classes)
            else:
                targets = tf.one_hot(tf.cast(targets, tf.int64),
                                     depth=num_classes,
                                     on_value=1.0,
                                     off_value=0.0)

        # Check one hot is correctly computed
        if targets.get_shape()[1] != num_classes:
            raise RuntimeError("Classification loss expects one hot encoding")

        return tf.to_float(targets)


class RegressionLoss(LossFn):
    """ Abstract class for losses in regression tasks """

    __metaclass__ = abc.ABCMeta

    def preprocess(self, logits, targets):
        return targets


class MixtureDensityLoss(RegressionLoss):

    """ Loss function for Mixture Density Networks """

    def __init__(self, distribution):
        """ Initializes a Mixture Density loss function
        Args:
            distribution: Distribution to use in the mixture
        """
        self.dist = distribution

    def check_compatibility(self):
        return False

    def loss(self, logits, targets):
        """ See base class """
        if has_method(self.dist, "optimized_loss"):
            return self.dist.optimized_loss(targets, logits)
        else:
            right = self.dist.pdf(targets, logits)
            # Weighted is a [N, K], which multiplies the weights and the
            # corresponding distributions
            weighted = tf.multiply(self.dist.weights(logits), right)
            # Sum over columns
            summed = tf.reduce_sum(weighted, axis=1)
            # Summed is a matrix [N]
            # Trick to avoid Nan
            safe = tf.maximum(tf.constant(1e-6), summed)
            return tf.reduce_mean(-tf.log(safe))

    def get_weight_term(self):
        return -1.0


class CrossEntropy(ClassificationLoss):
    """ Cross entropy for strict sparse classification
        (only one class is possible)
            e.g: groundtruth of instance as [0, 0, 1, 0]
    """

    def loss(self, logits, targets):
        return tf.nn.softmax_cross_entropy_with_logits(labels=targets,
                                                       logits=logits)


class RootMeanSquared(RegressionLoss):

    """ Computes the Mean Squared Error between model output
    and groundtruth """

    def loss(self, logits, targets):
        return tf.sqrt(tf.reduce_mean(tf.square(logits - targets)))


class MeanSquared(RegressionLoss):

    """ Mean squared error between prediction and label """

    def loss(self, logits, targets):
        return tf.square(logits - targets)


def smooth_neighbouring(targets, true_prob, num_classes):
    """ Converts the input target labels into one-hot vectors so the
    true label has probability true_prob and the probability remaining
    is equally distributed among the neighbouring labels
    """
    batch_size = tf.shape(targets)[0]

    # Set value to the true value in one-hot style
    one_hot = tf.one_hot(targets,
                         depth=num_classes,
                         on_value=true_prob,
                         off_value=0.0)

    # Matrix of size [batch_size, num_classes] where each row is the sequence:
    # {0, ... num_classes-1}
    ranges = tf.reshape(
        tf.tile(tf.range(0, num_classes), tf.stack([batch_size])),
        tf.stack([batch_size, num_classes])
    )

    # Matrix of size [batch_size, num_classes] where each row contains copies
    # of the true label
    labels = copy_columns(targets, num_classes)

    # Apply mask and get [batch_size x num_classes] matrix where each
    # position is the distance (in positions) to the true label within
    # the one-hot vector
    masked = tf.abs(tf.cast(ranges, tf.int32) - tf.cast(labels, tf.int32))

    # Get neighboring positions
    neighbors = tf.equal(masked, tf.constant(1))
    counts = tf.reduce_sum(tf.cast(neighbors, tf.int32), axis=[1])

    # Set neighboring values
    remaining = tf.add(tf.constant(1.0), tf.constant(-true_prob))
    # Counts is never 0 (at least more than one class in a classification
    # problem)
    neigh_value = tf.div(remaining, tf.cast(counts, tf.float32))
    neigh_values = copy_columns(neigh_value, num_classes)

    # Add neigbhoring values to original one hot
    return tf.multiply(neigh_values, tf.cast(neighbors, tf.float32)) + one_hot


def smooth_labels(targets, true_prob, num_classes):
    """ Converts input target labels into one-hot vectors where the true
    label has probability true_prob and the remaining probability is
    distributed among other classes
    """
    remaining = tf.add(tf.constant(1.0), tf.constant(-true_prob))
    # Since at least 2 classes must exist, division is never by 0
    off = tf.div(remaining, tf.add(tf.cast(num_classes, tf.float32),
                 tf.constant(-1.0)))
    return tf.one_hot(targets,
                      depth=num_classes,
                      on_value=true_prob,
                      off_value=off)
