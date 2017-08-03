
""" Test for smooth labeling in loss functions """

from widedeep.utils import get_logger
from widedeep.ops.distributions import GaussianMixture
from widedeep.ops.losses import MixtureDensityLoss
import numpy as np
import tensorflow as tf
from scipy.stats import norm

import unittest


class GaussianMixtureOptTestCae(unittest.TestCase):

    """ Test for the optimized Gaussian mixture density loss """

    def setUp(self):
        """ Prepare data """
        # Define data
        self.K = np.random.randint(2, 6)
        self.out_size = 1

        self.batch_size = np.random.randint(20, 100)
        self.logits_data = np.random.random((self.batch_size,
                                             self.K * (self.out_size + 2)))
        self.targets_data = np.asarray(np.random.random((self.batch_size,
                                                         self.out_size)))

        # Define Tensorflow placeholders
        self.pl_logits = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.K * (self.out_size + 2))
        )
        self.pl_targets = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.out_size)
        )

        self.logger = get_logger('data')

    def runTest(self):
        """ Run experiment """
        # Groundtruth value
        true_value = self.manual_loss(self.logits_data,
                                      self.targets_data,
                                      self.K)

        # Graph value
        graph_value = self.graph_loss(self.pl_logits, self.pl_targets, self.K)

        # Compute and compare
        with tf.Session() as sess:
            # Init all variables
            sess.run(tf.global_variables_initializer())

            # Get and log results
            graph_result = sess.run(graph_value,
                                    feed_dict={
                                        self.pl_logits: self.logits_data,
                                        self.pl_targets: self.targets_data
                                    })
            self.logger.info('True: {}'.format(true_value))
            self.logger.info('Output: {}'.format(graph_result))

            self.assertTrue(np.isclose(true_value, graph_result, rtol=0.05))

    def manual_loss(self, logits, targets, K):
        """ Computes the Gaussian Mixture Loss out of the graph computation """
        mixs, sigmas, means = logits[:, 0:K], logits[:, K:2*K], logits[:, 2*K:]
        mixs = np.exp(mixs)/np.sum(np.exp(mixs))  # Apply softmax
        sigmas = np.exp(sigmas)
        # Compute over all instances
        logexps = []
        for i in range(self.batch_size):
            sumexp = np.sum(
                [
                    mixs[:, k] *
                    norm.pdf(targets[i], means[i, k], sigmas[i, k])
                    for k in range(K)
                ]
            )
            logexps.append(np.log(sumexp))
        return -np.mean(logexps)

    @staticmethod
    def graph_loss(logits, targets, K):
        """ Computes the Gaussian Mixture loss using graph computation """
        mixture_gaussian = GaussianMixture(K)
        mg = MixtureDensityLoss(mixture_gaussian)
        return mg.compute_loss(logits, targets)


if __name__ == '__main__':
    unittest.main()
