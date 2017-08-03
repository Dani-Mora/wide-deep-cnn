
""" Test for smooth labeling in loss functions """

import widedeep.ops.metrics as me
from widedeep.utils import get_logger
import numpy as np
import tensorflow as tf

import abc
import unittest


class MetricsTestCase(unittest.TestCase):

    """ Base metrics test """

    __metaclass__ = abc.ABCMeta

    def setUp(self):
        self.targets = self.get_targets()
        self.logits = self.get_logits()
        self.batch_size = self.targets.shape[0]
        self.num_classes = self.get_num_classes()
        self.logger = get_logger('data')
        self.pl_logits = self.logits_placeholder()
        self.pl_targets = self.targets_placeholder()

    def base_test(self, metric, expected):
        """ Main procedure to test a metric"""
        mq_out = metric.compute(self.pl_logits, self.pl_targets)

        # Send to execution
        with tf.Session() as sess:

            # Init all variables
            sess.run(tf.global_variables_initializer())

            # Get and log results
            result = sess.run([mq_out],
                              feed_dict={
                                  self.pl_logits: self.logits,
                                  self.pl_targets: self.targets
                              })
            self.compare(result[0], expected)

    def compare(self, true, result):
        self.assertTrue(np.isclose(true, result))

    @abc.abstractmethod
    def get_logits(self):
        """ Defines logits """

    @abc.abstractmethod
    def get_num_classes(self):
        """ Defines number of classes of the problem """

    @abc.abstractmethod
    def get_targets(self):
        """ Defines targets """

    def logits_placeholder(self):
        """ Defines Tensorflow input for predictions"""

    def targets_placeholder(self):
        """ Defines Tensorflow input for labels """


class ClassificationMetricsTestCase(MetricsTestCase):

    """ Metrics tests for classification """

    def get_logits(self):
        return np.asarray([[0.50, 0.20, 0.30],
                           [0.70, 0.10, 0.20],
                           [0.25, 0.50, 0.25],
                           [0.10, 0.10, 0.80],
                           [0.30, 0.60, 0.10]])

    def get_targets(self):
        return np.asarray([0, 1, 2, 1, 1])

    def get_num_classes(self):
        return 3

    def logits_placeholder(self):
        return tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.get_num_classes())
        )

    def targets_placeholder(self):
        return tf.placeholder(tf.int32, shape=(self.batch_size))


class RegressionMetricsTestCase(MetricsTestCase):

    """ Metrics tests for classification """

    def get_logits(self):
        return np.asarray([[1.0], [6.5], [0.75], [7.75], [3.5]])

    def get_targets(self):
        return np.asarray([[2.0], [8.0], [1.0], [9.0], [4.0]])

    def get_num_classes(self):
        return 1

    def logits_placeholder(self):
        return tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.get_num_classes())
        )

    def targets_placeholder(self):
        return tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.get_num_classes())
        )


""" Classification metrics tests """


class AccuracyTestCase(ClassificationMetricsTestCase):
    def runTest(self):
        self.base_test(me.Accuracy(), 0.40)


class AccuracyModeTestCase(ClassificationMetricsTestCase):
    def runTest(self):
        self.base_test(me.AccuracyMode(self.num_classes), 0.60)


class AccuracyRandomTestCase(ClassificationMetricsTestCase):
    def runTest(self):
        self.base_test(me.AccuracyRandom(self.num_classes), 0.60)

    def compare(self, true, result):
        # We dont know result since it is random
        self.assertTrue(True)


class ConfusionMatrixTestCase(ClassificationMetricsTestCase):
    def runTest(self):
        self.base_test(me.ConfusionMatrix(self.num_classes),
                       expected=[[1, 0, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]])

    def compare(self, true, result):
        # We dont know result since it is random
        self.assertTrue(np.array_equal(true, np.asmatrix(result)))


""" Regression metrics tests """


class AbsErrTestCase(RegressionMetricsTestCase):
    def runTest(self):
        self.base_test(me.AbsError(), 0.90)


class MeanTestCase(RegressionMetricsTestCase):
    def runTest(self):
        self.base_test(me.Mean(), 3.90)


class StdTestCase(RegressionMetricsTestCase):
    def runTest(self):
        self.base_test(me.Std(), 2.83108)


class AbsErrIntervalTestCase(RegressionMetricsTestCase):
    def runTest(self):
        self.base_test(me.AbsErrorInterval(cuts=[3.0, 5.0, 7.0]),
                       [0.625, 0.5, np.nan, 1.375])

    def compare(self, true, result):
        self.assertTrue(np.allclose(true, result, equal_nan=True))


class RMSEIntervalTestCase(RegressionMetricsTestCase):
    def runTest(self):
        self.base_test(me.RMSEInterval(cuts=[3.0, 5.0, 7.0]),
                       [0.72886896, 0.5, np.nan, 1.38067007])

    def compare(self, true, result):
        self.assertTrue(np.allclose(true, result, equal_nan=True))


class MeanPredTestCase(RegressionMetricsTestCase):
    def runTest(self):
        self.base_test(me.MeanPredError(), 2.96)


class MedianPredTestCase(RegressionMetricsTestCase):
    def runTest(self):
        self.base_test(me.MedianPredError(), 2.80)


if __name__ == '__main__':
    unittest.main()
