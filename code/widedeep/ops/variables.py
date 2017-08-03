"""
Helper class to create variables in Tensorflow
"""

import tensorflow as tf

NON_SAVABLE_COL = 'not_saveable'


def variable(name,
             shape,
             dtype,
             initializer,
             collections=[],
             regularizer=None,
             trainable=True):
    """ Creates or retrieves existing variable to represent trainable weights
    Simplified version of the contrib convolution2d function extended for
    inference
    Args:
        name: Name of the variable
        shape: Shape (list) of the variable to create
        dtype: Type of the variable
        initializer: Initializer to compute.
        collections: Collections where variables should be added
        regularizer: Regularized function for the variable.
        trainable: Whether the value of the variable should be
            subject to training
    Returns
        Variable created/retrieved
    """
    # All variables are added to the variable collection
    col_set = build_collections(collections)
    return tf.get_variable(name,
                           shape=shape,
                           dtype=dtype,
                           initializer=initializer,
                           regularizer=regularizer,
                           trainable=trainable,
                           collections=col_set)


def auxiliar_variable(name,
                      shape,
                      dtype,
                      initializer,
                      collections=[],
                      regularizer=None):
    """ Creates or retrieves an existing variable that won't be stored in
    the checkpoints. This variable is not trainable. Parameters are further
    described in variable method"""
    return variable(name=name,
                    shape=shape,
                    dtype=dtype,
                    initializer=initializer,
                    collections=[NON_SAVABLE_COL] + collections,
                    regularizer=regularizer,
                    trainable=False)


def savable_variables():
    """ Returns the list of variables to save """
    all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    exclude = tf.get_collection(NON_SAVABLE_COL)
    excluded_names = [v.name for v in exclude]
    return [v for v in all if v.name not in excluded_names]


def build_collections(collections):
    """ Builds the input collections, adding the default collection graph
    ops.GraphKeys.VARIABLES and checks that collection names are unique """
    inp_cols = [] if collections is None else list(collections)
    cols = [tf.GraphKeys.GLOBAL_VARIABLES] + inp_cols
    return set(cols)  # Remove duplicates


def add_variable_summary(value, name):
    """ Sets summaries for the input variable """
    tf.summary.scalar(name="%s:fraction_of_zero_values" % name,
                      tensor=tf.nn.zero_fraction(value))
    tf.summary.histogram(name="%s:activation" % name, values=value)


def summarize_gradients(model_name, gradients):
    """ Adds histograms for gradients and gradient norms of the input
    gradients """

    def get_prefix(var):
        return model_name + '/' + var.name

    for gradient, variable in gradients:
        if gradient is not None:
            tf.summary.histogram(get_prefix(variable) + "/gradients", gradient)
            tf.summary.histogram(get_prefix(variable) + "/gradient_norm",
                                 tf.global_norm([gradient]))
