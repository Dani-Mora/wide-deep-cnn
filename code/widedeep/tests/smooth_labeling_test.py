
""" Test for smooth labeling in loss functions """

from widedeep.ops.losses import MeanSquared, CrossEntropy
from widedeep.utils import get_logger
import numpy as np
import tensorflow as tf

import unittest


class SmoothLabellingTestCase(unittest.TestCase):

    def setUp(self):

        self.targets = np.asarray([0, 1, 2, 1])
        self.batch_size = len(self.targets)
        self.num_classes = 3
        self.logits = np.random.random((self.batch_size, self.num_classes))

        self.logger = get_logger('data')

        # Define Tensorflow placeholder
        self.pl_logits = tf.placeholder(
            tf.float32, 
            shape=[None, self.num_classes]
        )
        self.pl_targets = tf.placeholder(tf.int32, shape=[None])

    def test_no_smooth_regression(self):
        mq = MeanSquared()
        mq_out = mq.preprocess(self.pl_logits, self.pl_targets)

        # Send to execution
        with tf.Session() as sess:

            # Init all variables
            sess.run(tf.global_variables_initializer())

            # Get and log results
            result = sess.run(
                [mq_out],
                feed_dict={
                    self.pl_logits: self.logits,
                    self.pl_targets: self.targets
                }
            )

            self.logger.info(
                'Regression preprocessing of {} is {}'
                .format(self.targets, result[0])
            )

            self.assertTrue(np.array_equal(result[0], self.targets))

    def test_one_hot(self):

        ce = CrossEntropy()
        ce_out = ce.preprocess(self.pl_logits, self.pl_targets)

        # Send to execution
        with tf.Session() as sess:
            # Init all variables
            sess.run(tf.global_variables_initializer())

            # Get and log results
            result = sess.run(
                [ce_out],
                feed_dict={
                    self.pl_logits: self.logits,
                    self.pl_targets: self.targets
                }
            )

            self.logger.info('One-hot of {} is {}'
                             .format(self.targets, result[0]))

            # Check true values and computed ones agree
            true_one_hot = np.asmatrix(
                [
                    [1.0, 0, 0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0]
                ]
            )

            self.assertTrue(np.array_equal(result[0], true_one_hot))

    def test_smooth_labelling(self):

        on = 0.6
        off = (1.0 - on)/(self.num_classes-1.0)
        ce_smooth = CrossEntropy(smooth_label=on)
        ce_smooth_out = ce_smooth.preprocess(self.pl_logits, self.pl_targets)

        # Send to execution
        with tf.Session() as sess:

            # Init all variables
            sess.run(tf.global_variables_initializer())

            # Get and log results
            result = sess.run(
                [ce_smooth_out],
                feed_dict={
                    self.pl_logits: self.logits, 
                    self.pl_targets: self.targets
                }
            )

            self.logger.info('Smooth labelling of {} is {}'
                             .format(self.targets, result[0]))

            # Check true values and computed ones agree
            smoothed = np.asmatrix(
                [
                    [on, off, off],
                    [off, on, off],
                    [off, off, on],
                    [off, on, off]
                ]
            )
            np.testing.assert_array_almost_equal(smoothed, result[0])

    def test_smooth_neighboring_labelling(self):

        on = 0.6
        neigh = (1.0 - on) / 2.0
        ce_neigh = CrossEntropy(smooth_neigh=on)
        ce_neigh_out = ce_neigh.preprocess(self.pl_logits, self.pl_targets)

        # Send to execution
        with tf.Session() as sess:
            # Init all variables
            sess.run(tf.global_variables_initializer())

            # Get and log results
            result = sess.run(
                [ce_neigh_out],
                feed_dict={
                    self.pl_logits: self.logits,
                    self.pl_targets: self.targets
                }
            )

            self.logger.info('Smooth neighboring labelling of {} is {}'
                             .format(self.targets, result[0]))

            # Check true values and computed ones agree
            smoothed = np.asmatrix(
                [
                    [on, neigh*2.0, 0.0],
                    [neigh, on, neigh],
                    [0.0, neigh*2.0, on],
                    [neigh, on, neigh]
                ]
            )
            np.testing.assert_array_almost_equal(smoothed, result[0])


if __name__ == '__main__':
    unittest.main()
