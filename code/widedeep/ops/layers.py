"""
Helper class for creating network layers
"""

from widedeep.ops.variables import variable, add_variable_summary
from widedeep.utils import get_logger

from tensorflow.contrib import layers
import tensorflow as tf

logger = get_logger('data')


def conv_layer(inputs,
               is_training,
               output_size,
               kernel_size,
               stride=1,
               group=1,
               padding='SAME',
               activation_fn=tf.nn.relu,
               lrn=None,
               batch_norm=False,
               pool=None,    # max_pool2d, avg_pool2d
               pool_size=2,
               pool_stride=3,
               pool_padding='SAME',
               summary=True,
               weights_initializer=layers.xavier_initializer(),
               biases_initializer=tf.constant_initializer(0.1),
               weight_collections=None,
               ops_collections=None,
               scope=None):  # name of the layer
    """ Creates a convolutional layer according to the given parameters
    Args:
        inputs: Input of the convolutional layer.
        is_training: 1D boolean tensor indicating whether training is
            being done or not
        output_size: Output dimension of the convolutional layer.
        kernel_size: Side of the kernel (squared) to use.
        stride: Stride to use in the convolution.
        group: Number of groups to use for convolving. Default is 1.
            This is used for a specific case of the Alexnet network definition.
        padding: Type of padding to apply ('SAME' or 'VALID').
            Default is 'SAME'.
        activation_fn: Activation function to apply after the convolution.
            Set to None for identify
        lrn: Parameters of the Local Response Normalization:
            [alpha, beta, bias, radius]. Set to None to disable them.
        batch_norm: Whether to apply batch normalization or not.
            By default it is False.
        pool: Pooling function to apply (e.g. max_pool2d).
            Set to None to disable it.
        pool_size: Size of the pooling, if any.
        pool_stride: Stride of the pooling, if any.
        pool_padding: Padding of the pooling layer ('SAME' or 'VALID').
            Default is 'SAME'.
        summary: Whether to add summary of the weight matrix value.
        weights_initializer: initializer of the weights. By default,
            it is Xavier initializer
        biases_initializer: Biases initializer.
            By default it is initialized to zeros.
        weight_collections: Collections where to add the trainable variables.
            By default added to GraphKeys.VARIABLES.
        ops_collections: Collections where to add the update operations,
            if any. By default added to GraphKeys.UPDATE_OPS.
        scope: Name of the scope where to add the variables.
    """

    def conv_fn(inp, w):
        return tf.nn.conv2d(inp,
                            w,
                            strides=[1, stride, stride, 1],
                            padding=padding)

    if lrn is not None and batch_norm is True:
        logger.warn("Using batch norm and together")

    with tf.variable_scope(scope) as sc:

        # Let's check that the input and output channels are
        # divisible by the group parameter
        input_size = inputs.get_shape().as_list()[-1]
        assert input_size % group == 0
        assert output_size % group == 0

        # Convolutional weights
        weights_shape = [kernel_size,
                         kernel_size,
                         tf.divide(input_size, group),
                         output_size]
        weights = variable('weights', shape=weights_shape,
                           dtype=inputs.dtype.base_dtype,
                           initializer=weights_initializer,
                           collections=weight_collections,
                           trainable=True)

        if group == 1:
            outputs = conv_fn(inputs, weights)
        else:
            # Split both inputs and kernels into groups
            input_groups = tf.split(axis=3,
                                    num_or_size_splits=group,
                                    value=inputs)
            kernel_groups = tf.split(axis=3,
                                     num_or_size_splits=group,
                                     value=weights)
            # Convolve groups independently and concatenate them
            output_groups = [conv_fn(i, k)
                             for i, k in zip(input_groups, kernel_groups)]
            outputs = tf.concat(values=output_groups, axis=3)

        # Convolutional biases: conv(x, w) + b
        biases = variable('biases', shape=[output_size, ],
                          dtype=inputs.dtype.base_dtype,
                          initializer=biases_initializer,
                          collections=weight_collections,
                          trainable=True)

        outputs = tf.nn.bias_add(outputs, biases)

        # Batch normalization
        if batch_norm:
            logger.info('Added batch normalization for {}'.format(sc.name))
            outputs = tf.contrib.layers.batch_norm(
                outputs,
                is_training=is_training,
                center=True,
                scale=True,
                variables_collections=weight_collections,
                updates_collections=ops_collections
            )

        if summary is True:
            add_variable_summary(name=sc.name + '_preactivation',
                                 value=outputs)

        # Apply non-linearity: f(conv(x, w) + b)
            outputs = activation_fn(outputs)  # Non-linearity

        # Apply pooling layer, if requested: pool(f(conv(x,w) + b))
        if pool is not None:
            outputs = pool(outputs,
                           ksize=[1, pool_size, pool_size, 1],
                           strides=[1, pool_stride, pool_stride, 1],
                           padding=pool_padding)  # Pooling

        # End with LRN, if requested: lrn(pool(f(conv(x,w) + b)))
        if lrn is not None:
            alpha, beta, bias, radius = lrn
            outputs = tf.nn.local_response_normalization(outputs,
                                                         alpha=alpha,
                                                         beta=beta,
                                                         bias=bias,
                                                         depth_radius=radius)
        if summary is True:
            add_variable_summary(name=sc.name + '_postactivation',
                                 value=outputs)

        return outputs


def cnnlayer_from_config(layer,
                         net,
                         weight_collections,
                         ops_collections,
                         is_training):
    """ Returns the graph operations for the corresponding input cnn layer
    Args:
        layer: CNNLayer configuration
        net: CNNLayer input
        weight_collections: Collection where to add the trainable variables.
            By default added to GraphKeys.VARIABLES.
        ops_collections: Collections where to add the update operations,
            if any. By default added to GraphKeys.UPDATE_OPS
        is_training: 1D boolean tensor indicating whether training is being
            done or not
    """
    activation = get_activation_func(layer.activation)
    lrn = [layer.alpha, layer.beta, layer.bias, layer.radius] if layer.lrn else None  # noqa
    pool = get_pooling(layer.pool) if layer.pool is not None else None
    return conv_layer(inputs=net,
                      is_training=is_training,
                      output_size=layer.maps,
                      kernel_size=layer.ksize,
                      stride=layer.stride,
                      group=layer.group,
                      padding=layer.padding.upper(),
                      activation_fn=activation,
                      lrn=lrn,
                      batch_norm=layer.batch_norm,
                      pool=pool,
                      pool_size=layer.pool_size,
                      pool_stride=layer.pool_stride,
                      pool_padding=layer.pool_padding.upper(),
                      weight_collections=weight_collections,
                      ops_collections=ops_collections,
                      scope=layer.name)  # name of the layer


def input_layer(inputs, features, weight_collections, scope):
    """ Reads input features into valid tensors
    Args:
        inputs: Input dictionary containing the data.
        features: List of columns to read.
        weight_collections: Collection where to add the trainable variables.
            By default added to GraphKeys.VARIABLES.
        scope: Name of the scope where to add the variables.
    Returns:
        Processed tensors.
    """
    net = layers.input_from_feature_columns(
        inputs,
        features,
        weight_collections=weight_collections,
        scope=scope
    )
    return net


def linear_layer(inputs, features, outputs, weight_collections, scope):
    """ Generates an output by multiplying each feature value by a weight
    Args:
        inputs: Input dictionary containing the data.
        features: List of columns to read.
        outputs: Number of outputs/classes.
        weight_collections: Collection where to add the trainable variables.
            By default added to GraphKeys.VARIABLES.
        scope: Name of the scope where to add the variables.
    Returns:
        Output of the linear model.
    """
    with tf.variable_scope(scope) as sc:
        logits, _, _ = layers.weighted_sum_from_feature_columns(
            columns_to_tensors=inputs,
            feature_columns=features,
            num_outputs=outputs,
            weight_collections=weight_collections,
            scope=sc)
        return logits


def fully_connected(inputs,
                    hidden_units,
                    is_training,
                    dropout=None,
                    activation_fn=tf.nn.relu,
                    batch_norm=False,
                    summary=True,
                    weights_initializer=layers.xavier_initializer(),
                    biases_initializer=tf.constant_initializer(0.1),
                    weight_collections=None,
                    ops_collections=None,
                    scope=None):  # name of the layer
    """ Creates a convolutional layer according to the given parameters
    Args:
        inputs: Input of the fully connected layer
        hidden_units: Number of output units
        is_training: 1D boolean tensor indicating whether training is
            being done or not.
        dropout: Ratio of units to deactivate randomly.
            Set to None or 0.0 to disable it. Only used for training.
        activation_fn: Activation function to apply. Set to None for identify
        batch_norm: Whether to apply batch normalization or not.
        summary: Whether to add summary of the weight matrix value
        weights_initializer: initializer of the weights. By default,
            it is Xavier initializer.
        biases_initializer: Biases initializer. By default it is initialized
            to zeros.
        weight_collections: Collection where to add the trainable variables.
            By default added to GraphKeys.VARIABLES.
        ops_collections: Collections where to add the update operations,
            if any. By default added to GraphKeys.UPDATE_OPS.
        scope: Name of the scope where to add the variables
    """
    # Dropout checker
    if dropout is not None and (dropout < 0.0 or dropout > 1.0):
        raise ValueError('Dropout out of range! Must be in [0, 1]')

    with tf.variable_scope(scope) as sc:

        # Define FC weights
        input_size = inputs.get_shape()[-1]
        weights_shape = [input_size, hidden_units]
        W = variable('weights', shape=weights_shape,
                     dtype=inputs.dtype.base_dtype,
                     initializer=weights_initializer,
                     collections=weight_collections,
                     trainable=True)

        # Define biases
        b = variable('biases', shape=[hidden_units, ],
                     dtype=inputs.dtype.base_dtype,
                     initializer=biases_initializer,
                     collections=weight_collections,
                     trainable=True)

        # Apply weight product and bias addition
        net = tf.nn.bias_add(tf.matmul(inputs, W), b)

        # Batch normalization
        if batch_norm is True:
            net = tf.contrib.layers.batch_norm(
                net,
                is_training=is_training,
                center=True,
                scale=True,
                variables_collections=weight_collections,
                updates_collections=ops_collections
            )

        # Summary before activation
        if summary is True:
            add_variable_summary(name=sc.name + '_preactivation',
                                 value=net)

        # Apply activation function
        net = activation_fn(net)

        # Apply dropout
        if dropout is not None:
            net = tf.cond(is_training,
                          lambda: tf.nn.dropout(net, keep_prob=1.0-dropout),
                          lambda: tf.identity(net))

        # Summary after activation
        if summary is True:
            add_variable_summary(name=sc.name + '_postactivation',
                                 value=net)

        return net


def fclayer_from_config(layer,
                        net,
                        weight_collections,
                        ops_collections,
                        is_training):
    """ Returns the graph operations for the corresponding input fully
    connected layer
    Args:
        layer: FullyCLayer configuration
        net: FullyCLayer input
        weight_collections: Collections where to include the fully connected
            layer variables
        ops_collections: Collections where to add the update operations,
            if any. By default added to GraphKeys.UPDATE_OPS.
        is_training: 1D boolean tensor indicating whether training
            is being done or not
    """
    activation = get_activation_func(layer.activation)
    return fully_connected(inputs=net,
                           hidden_units=layer.hidden,
                           dropout=layer.dropout,
                           is_training=is_training,
                           activation_fn=activation,
                           batch_norm=layer.batch_norm,
                           weight_collections=weight_collections,
                           ops_collections=ops_collections,
                           scope=layer.name)  # name of the layer


def get_activation_func(activation_label):
    """ Returns the activation function given the label
    Args:
        activation_label: Name of the function
    """
    if activation_label == 'sigmoid':
        return tf.nn.sigmoid
    elif activation_label == 'identity':
        return tf.identity
    elif activation_label == 'relu':
        return tf.nn.relu
    elif activation_label == 'tanh':
        return tf.nn.tanh
    else:
        raise ValueError('Unknown activation function %s'
                         % activation_label)


def get_pooling(pooling_type):
    """ Returns the pooling function given the label
    Args:
        pooling_type: Name of the pooling function
    """
    if pooling_type == 'max2d':
        return tf.nn.max_pool
    elif pooling_type == 'avg2d':
        return tf.nn.avg_pool
    else:
        raise ValueError('Unknown pooling function %s'
                         % pooling_type)
