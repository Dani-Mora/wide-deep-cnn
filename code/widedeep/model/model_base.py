import widedeep.ops.layers as la
from widedeep.utils import get_logger
from widedeep.ops.variables import summarize_gradients
from widedeep.model.network_conf import CNNConfig, MLPConfig
from widedeep.ops.losses import Optimizers
from tensorflow.contrib import layers

import abc
import numpy as np
import math
import tensorflow as tf


logger = get_logger('model')


class Model(object):
    """ Class that represents a trainable model from Tensorflow.
    All models add an extra fully connected at the end of the process in
    order to match the number of classes. This means that this last layer
    must NOT be indicated in the configuration """

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, columns_list, optimizer=None, initial_lr=None,
                 decay_steps=None, decay_rate=None, weights=None,
                 excluded_layers=[], **optimizer_args):
        """
        Args:
            name: Scope name of the model
            columns_list: Name of the columns the model uses as input
            optimizer: Optimizer class to use (losses.py).
                Set to None for default.
            initial_lr: Initial learning rate for the model. If None, will use
                the default one.
            decay_steps: Steps at which we decrease the learning rate.
                Set to None to disable
            decay_rate: Ratio at which we decrease the learning rate.
                Set to None to disable
            weights: Path to the file containing pretrained weights.
                If None and no pretrained model is loaded, the model will
                train weights from scratch.
            excluded_layers: Name of the layers we do not want to initialize
                from weights. This is only used when weights is not None.
            optimizer_args: Arguments of the provided optimizer
        """
        self._name = name
        self._feature_columns = columns_list
        self._weight_collection_name = 'trainable_' + name
        self._update_collection_name = 'updates_' + name
        self._optimizer = optimizer
        self._initial_lr = initial_lr
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._weights = weights
        self._excluded = excluded_layers
        self._opt_args = optimizer_args

        if (self._decay_rate and not self._decay_steps) or \
           (self._decay_steps and not self._decay_rate):
            raise ValueError('Decay steps and rate must be both ' +
                             'provided (or none of them)')

    def get_name(self):
        return self._name

    def get_logits(self, features, is_training, outputs):
        """ Returns the output of the model """
        with tf.variable_scope(self._name):
            logits = self.get_model_logits(features,
                                           is_training=is_training,
                                           outputs=outputs)
            return logits

    def warm_start(self):
        """ Assigns the weights provided, if any
        Returns:
            List of operations to assign the initial weights.
            Empty list if no path provided
        """
        if self._weights is not None:
            logger.info('Loading weight initialization for model %s'
                        % (self._name) +
                        ' from weights in %s' % (self._weights))
            return self.assign_weights(self._weights)
        return []

    def get_training_ops(self, step, loss, clip_grad=None):
        """
        Args:
            step: Variable pointing to the global step
            loss: Loss to use for computing the gradients
            clip_grad: If not None, gradients are clipped into interval
                [-clip_grad, clip_grad]
        Returns:
            Update operations to perform at each step """
        my_vars = self.get_parameters()
        optimizer = self._get_optimizer()
        # Get base learning rate
        if self._initial_lr is None:
            lr = self._get_default_lr()
        else:
            lr = self._initial_lr
        # Set decay, if requested
        if self._decay_rate is not None:
            lr = tf.train.exponential_decay(learning_rate=lr,
                                            global_step=step,
                                            decay_rate=self._decay_rate,
                                            decay_steps=self._decay_steps)
        # Track learning rate
        tf.summary.scalar(name="learning_rate_" + self._name, tensor=lr)

        # Create optimizer
        logger.info('Using %s optimizer' % optimizer)
        opt = optimizer(learning_rate=lr, use_locking=True, **self._opt_args)

        # Compute gradients
        gradients = opt.compute_gradients(loss, var_list=my_vars)

        # Gradient clipping
        if clip_grad is not None:
            logger.info("Clipping gradients into [-%f, %f]"
                        % (clip_grad, clip_grad))
            gradients = [(tf.clip_by_norm(grad, clip_grad), var)
                         for grad, var in gradients]

        # Track gradient information
        summarize_gradients(self._name, gradients)

        # Apply updates operation
        grad_updates = opt.apply_gradients(gradients)

        # We could use optimizer.minimize() directly but using 2 steps allows
        # us to track the gradients
        return grad_updates

    def _get_feature_columns(self):
        return self._feature_columns

    def get_parameters(self):
        """ Returns the trainable values of the model """
        return tf.get_collection(self._weight_collection_name)

    def get_extra_operations(self):
        """ Returns model-specific update operations """
        return tf.get_collection(self._update_collection_name)

    def _get_optimizer(self):
        if self._optimizer is None:
            logger.info('No optimizer given. Setting default optimizer ...')
            optimizer = self._get_default_optimizer()
        else:
            optimizer = self._optimizer
        return optimizer

    def assign_weights(self, path):
        """ Assigns the weights in the path to the model.
        Make sure weights are named the same way they are defined in the
        model. If any name provided at construction, some layers can be
        excluded and be initialized from scratch """
        assign_ops = []
        weights = np.load(path, encoding='latin1').item()
        with tf.variable_scope(self.get_name(), reuse=True):
            scopes = sorted(weights.keys())
            # Iterate through layers
            for sc in scopes:

                if sc not in self._excluded:

                    # Read current layer content
                    current = weights[sc]

                    # If list, translate into dictionary
                    if isinstance(current, list):
                        current = {'weights': current[0],
                                   'biases': current[1]}

                    # Ensure it is readable
                    if not isinstance(current, dict):
                        raise RuntimeError('Layer weights must be provided '
                                           + 'as a dictionary')

                    # Assign weights and biases for the current layer
                    for param_name, data in current.items():
                        var_name = sc + '/' + param_name
                        assign_ops.append(tf.get_variable(var_name)
                                            .assign(data))
                        logger.info("Warm start of {}({}) from {}"
                                    .format(param_name, sc, path))

        return assign_ops

    @abc.abstractmethod
    def get_model_logits(self, features, is_training, outputs):
        """ Returns the output of the model
        Args:
            features: input set of features (dictionary of valid columns)
            is_training: 1D boolean tensor indicating whether we are
                optimizing the model or not (evaluation)
            outputs: Size of the output layer
        """

    @abc.abstractmethod
    def _get_default_optimizer(self):
        """ Returns the default optimizer for the model """

    @abc.abstractmethod
    def _get_default_lr(self):
        """ Returns the default learning rate the model """


class LinearModel(Model):
    """ Class that represents a model that linearly combines categorical
    features using an mapping between each of the possible values/hashes
    as weights """

    def __init__(self, name, columns_list, optimizer=None, initial_lr=None,
                 decay_steps=None, decay_rate=None, weights=None,
                 excluded_layers=[], **optimizer_args):
        """ See parent's class constructor """
        super(LinearModel, self).__init__(name, columns_list, optimizer,
                                          initial_lr, decay_steps,
                                          decay_rate, weights,
                                          excluded_layers, **optimizer_args)

    def get_model_logits(self, features, is_training, outputs):
        return la.linear_layer(
            inputs=features,
            features=self._get_feature_columns(),
            outputs=outputs,
            weight_collections=[self._weight_collection_name],
            scope='linear'
        )

    def _get_default_optimizer(self, optimizer_name=None):
        return Optimizers.Ftrl

    def _get_default_lr(self):
        feature_num = len(self._get_feature_columns())
        default_learning_rate = 1. / math.sqrt(feature_num)
        return min(0.2, default_learning_rate)


class MLP(Model):
    """ Class that represents a Multilayer Perceptron using input numerical
    features """

    def __init__(self, name, columns_list, layers, optimizer=None,
                 initial_lr=None, decay_steps=None, decay_rate=None,
                 weights=None, excluded_layers=[], **optimizer_args):
        """ Initializes a Multilayer Perceptron Model
        Args:
            layers: Network configuration file containing the layer's
                definitions
            See base class for other arguments
        """
        super(MLP, self).__init__(name, columns_list, optimizer, initial_lr,
                                  decay_steps, decay_rate, weights,
                                  excluded_layers, **optimizer_args)
        # Read cnn configuration
        config = MLPConfig()
        config.read(layers)
        self.config = config

    def get_model_logits(self, features, is_training, outputs):
        """ See base class """

        # Load inputs
        net = la.input_layer(
            inputs=features,
            features=self._get_feature_columns(),
            weight_collections=[self._weight_collection_name],
            scope='input_from_columns'
        )

        # Fully connected layers
        for fc_id, layer in enumerate(self.config.hidden_layers):
            net = la.fclayer_from_config(
                layer=layer,
                net=net,
                weight_collections=[self._weight_collection_name],
                ops_collections=[self._update_collection_name],
                is_training=is_training
            )

        # Output layer
        logits = la.fully_connected(
            net,
            hidden_units=outputs,
            is_training=is_training,
            dropout=None,
            activation_fn=la.get_activation_func('identity'),
            weight_collections=[self._weight_collection_name],
            ops_collections=[self._update_collection_name],
            scope='mlp_logits'
        )

        return logits

    def _get_default_optimizer(self, optimizer_name=None):
        return Optimizers.SGD

    def _get_default_lr(self):
        return 0.1


class CNNModel(Model):
    """ Class that represents a Convolutional Neural Network model """

    def __init__(self, name, columns_list, cnn_layers, optimizer=None,
                 initial_lr=None, decay_steps=None, decay_rate=None,
                 weights=None, excluded_layers=[], **optimizer_args):

        """ Initializes a Convolutional Model
        Args:
            cnn_layers: Network configuration file containing the
                layer's definitions
            See base class for other arguments
        """
        super(CNNModel, self).__init__(name, columns_list, optimizer,
                                       initial_lr, decay_steps, decay_rate,
                                       weights, excluded_layers,
                                       **optimizer_args)
        # Read cnn configuration
        config = CNNConfig()
        config.read(cnn_layers)
        self.config = config
        self.layers = {}
        self.last_hidden = None

    def get_model_logits(self, features, is_training, outputs):
        """See base class."""

        # Load inputs
        net = la.input_layer(inputs=features,
                             features=self._get_feature_columns(),
                             weight_collections=[self._weight_collection_name],
                             scope='input_from_columns')

        # Store summary for current model images
        tf.summary.image(name='cnn_preview', tensor=net, max_outputs=3)

        # Conv layers
        for conv_id, layer in enumerate(self.config.conv_layers):
            net = la.cnnlayer_from_config(
                layer=layer,
                net=net,
                weight_collections=[self._weight_collection_name],
                ops_collections=[self._update_collection_name],
                is_training=is_training
            )
            self.layers[layer.name] = net

        # Flatten so it fits the fully connected layers
        net = layers.flatten(net)

        # Fully connected layers
        for fc_id, layer in enumerate(self.config.full_layers):
            net = la.fclayer_from_config(
                layer=layer,
                net=net,
                weight_collections=[self._weight_collection_name],
                ops_collections=[self._update_collection_name],
                is_training=is_training
            )
            self.layers[layer.name] = net

        # Store last hidden for debugging purposes
        self.last_hidden = net

        # Output layer
        logits = la.fully_connected(
            net,
            hidden_units=outputs,
            is_training=is_training,
            dropout=None,
            activation_fn=la.get_activation_func('identity'),
            weight_collections=[self._weight_collection_name],
            ops_collections=[self._update_collection_name],
            scope='cnn_logits'
        )

        return logits

    def _get_default_optimizer(self, optimizer_name=None):
        return Optimizers.Adagrad

    def _get_default_lr(self):
        return 0.01
