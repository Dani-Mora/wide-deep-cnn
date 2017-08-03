
""" Class for common distributions to use with Mixture Density Networks """

import abc
import tensorflow as tf
import numpy as np


class Mixture(object):

    """ Mixture base class. We assume we only have one output """

    __metaclass__ = abc.ABCMeta

    def __init__(self, components):
        self.components = components

    def pdf(self, x, logits, **kwargs):
        """ Wrapper for the Probability Density Function """
        return self._pdf(x, *self.logits_to_params(logits), **kwargs)

    def sample(self, logits, **kwargs):
        """ Wrapper for the sample function """
        return self._sample(*self.logits_to_params(logits), **kwargs)

    @abc.abstractmethod
    def _pdf(self, x, *params, **kwargs):
        """ Probability density function
        Args:
            x: Input data
            args: Arguments of the specific distribution
        """

    @abc.abstractmethod
    def _sample(self, *params, **kwargs):
        """ Samples a prediction given the model outputs.
        Args:
            logits: Network output
            args: Arguments of the specific distribution
        """

    @abc.abstractmethod
    def weights(self, logits):
        """ Returns the weight for each mixture component """

    @abc.abstractmethod
    def logits_to_params(self, logits):
        """ Separates the network outputs into the corresponding parameters """

    @abc.abstractmethod
    def outputs(self, output_size):
        """ Returns the number of outputs of the mixture
         Args:
            output_size: Size of the output itself
        """


class GaussianMixture(Mixture):

    """ Normal distribution """

    def __init__(self, components):
        super(GaussianMixture, self).__init__(components)

    def _pdf(self, x, *args, **kwargs):
        """  We assume that the model is heteroscedastic: covariance matrix
        is diagonal.
        TODO: we could extend this to non-diagonal covariance matrices
        """
        return self._gaussian_pdf(x, *args)

    def _gaussian_pdf(self, x, mixings, sigma, mean):
        """ Wrapper for Gaussian PDF """
        variance = tf.square(sigma)
        output_size = tf.cast(tf.shape(mean)[1], tf.float32)
        # Left: 1/sqrt(pi * 2 * variance) [N, K]
        left = tf.reciprocal(tf.pow(2*np.pi, output_size/2.0) *
                             tf.pow(sigma, output_size))
        # Exponent: e^[-(x-mu)^2/(2var)]. [N, K]
        right = tf.exp(-tf.divide(tf.square(x - mean), 2 * variance))
        return tf.multiply(left, right)

    def _sample(self, *params, **kwargs):
        """ Returns the mean of the most probable Gaussian distribution """
        # Get identifier of the most probable component
        mixings, sigma, mean = params
        batch_size = mixings.get_shape()[0]
        id_mix = tf.cast(tf.argmax(mixings, axis=1), tf.int32)

        # Extracted from https://github.com/tensorflow/tensorflow/issues/418
        # Get mean of corresponding component
        sample = tf.gather(
            params=tf.reshape(mean, [-1]),
            indices=tf.range(batch_size) * tf.shape(mean)[1] + id_mix
        )

        # Small workaround
        if sample.get_shape().ndims < 2:
            sample = tf.expand_dims(sample, axis=1)

        return sample

    def optimized_loss(self, targets, logits):
        """ Function that computes the loss of a mixture density network
        in a way that it handles underflow and overflow and avoids unstable
        behaviors """
        # Obtain parameters
        mixings, sigma, mean = self.logits_to_params(logits)
        output_size = tf.cast(tf.shape(targets)[1], tf.float32)
        variance = tf.square(sigma)
        # Convert expressions into exponent-based terms
        mixings_exp = tf.log(mixings)
        # By properties of logarithm we can simplify the original expression
        # log(x/y) = log(x) - log(y), log(xy) = log(x) + log(y), log(1) = 0
        sqrt_exp = - output_size * (0.5 * tf.log(2*np.pi) + tf.log(sigma))
        gaussian_exp = -tf.divide(tf.square(targets - mean), 2 * variance)
        exponent = mixings_exp + sqrt_exp + gaussian_exp
        # Use optimized logsumexp function to control underflow/overflow
        return tf.reduce_logsumexp(exponent, axis=1)

    def weights(self, logits):
        return self.logits_to_params(logits)[0]

    def logits_to_params(self, logits):
        # Affects to split and mean computation (fixed, only substraction)
        mixings = logits[:, 0:self.components]
        sigma = logits[:, self.components:self.components*2]
        mean = logits[:, self.components*2:]
        return tf.nn.softmax(mixings), tf.exp(sigma), mean

    def outputs(self, output_size):
        return self.components * (output_size + 2)
