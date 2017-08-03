
""" Test to check that the defined architecture works for a very simple dataset such as Diabetes UCI """

from protodata.datasets.scikit_dataset import DiabetesSettings
from protodata.datasets import Datasets
from protodata.utils import get_data_location
from protodata.data_ops import TrainMode, DataMode

from widedeep.model.model_base import LinearModel, MLP
from widedeep.model.joint_model import JointRegressor
from widedeep.ops.losses import Optimizers, MeanSquared
import widedeep.utils as ut
import widedeep.ops.metrics as me


import tensorflow as tf

logger = ut.get_logger('data')

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    "data_location",
    get_data_location(Datasets.DIABETES),
    "Where data is stored"
)

flags.DEFINE_integer(
    "batch_size",
    32,
    "Batch size to use."
)

flags.DEFINE_string(
    "network",
    ut.NetworkModels.MLP,
    "Network to use for MLP, if used"
)

flags.DEFINE_integer(
    "summaries",
    50,
    "Steps between summaries."
)

flags.DEFINE_integer(
    "checkpoints",
    100,
    "Steps between model checkpoints."
)

flags.DEFINE_integer(
    "steps",
    200,
    "Steps to train."
)

flags.DEFINE_float(
    "gpu_frac",
    0.70,
    "Percentage of GPU memory to use."
)

# High levels of gpu_frac can lead to Error:
# failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED
flags.DEFINE_string(
    "mode",
    TrainMode.DEEP,
    "Architecture to use"
)

flags.DEFINE_bool(
    "training",
    True,
    "Execution mode"
)

# Linear parameters
flags.DEFINE_float(
    "linear_initial_lr",
    0.01,
    "Initial learning rate for the linear model."
)

flags.DEFINE_integer(
    "linear_decay_steps",
    None,
    "Steps at which learning rate decreases for the linear model."
)

flags.DEFINE_float(
    "linear_decay_rate",
    None,
    "Decrease rate of the learning rate for the linear model."
)

# Regularization
flags.DEFINE_float(
    "l1_regularization",
    None,
    "L1 regularization for the loss. Set to None to disable"
)

flags.DEFINE_float(
    "l2_regularization",
    None,
    "L2 regularization for the loss. Set to None to disable"
)

# Gradient clipping
flags.DEFINE_float(
    "gradient_clip",
    None,
    "If not None, value to use for clipping the gradient"
)

# MLP parameters
flags.DEFINE_float(
    "mlp_initial_lr",
    0.001,
    "Initial learning rate for the MLP model."
)

flags.DEFINE_integer(
    "mlp_decay_steps",
    10000,
    "Steps at which learning rate decreases for the MLP model."
)

flags.DEFINE_float(
    "mlp_decay_rate",
    0.5,
    "Decrease rate of the learning rate for the MLP model."
)

flags.DEFINE_string(
    "mlp_network",
    ut.NetworkModels.MLP,
    "Network to use for the MLP"
)


if __name__ == '__main__':

    # Select Airbnb dataset
    dataset = DiabetesSettings(dataset_location=FLAGS.data_location)

    # Define columns
    wide = dataset.get_wide_columns()

    logger.info('Using columns {}'.format(wide))

    linear_model = LinearModel('linear',
                               columns_list=wide,
                               optimizer=Optimizers.SGD,
                               initial_lr=FLAGS.linear_initial_lr,
                               decay_steps=FLAGS.linear_decay_steps,
                               decay_rate=FLAGS.linear_decay_rate)

    mlp_model = MLP('mlp',
                    columns_list=wide,
                    layers=ut.get_network_definition(FLAGS.mlp_network),
                    optimizer=Optimizers.SGD,
                    initial_lr=FLAGS.mlp_initial_lr,
                    decay_steps=FLAGS.mlp_decay_steps,
                    decay_rate=FLAGS.mlp_decay_rate)

    # Set models according to settings
    if FLAGS.mode == TrainMode.WIDE:
        models = [linear_model]
    elif FLAGS.mode == TrainMode.DEEP:
        models = [mlp_model]
    elif FLAGS.mode == TrainMode.WIDE_AND_DEEP:
        models = [linear_model, mlp_model]
    else:
        raise ValueError('Unsupported option in Diabetes training %s'
                         % FLAGS.mode)

    # Create model
    joint = JointRegressor(ut.get_default_output(),
                           models=models,
                           outputs=dataset.get_num_classes(),
                           l1_reg=FLAGS.l1_regularization,
                           l2_reg=FLAGS.l2_regularization,
                           loss_fn=MeanSquared(),
                           clip_gradient=FLAGS.gradient_clip)

    # Define metrics
    metrics = [
        me.AbsError(),
        me.MeanPredError(),
        me.MedianPredError(),
        me.Mean(),
        me.Std(),
        me.AbsErrorInterval(cuts=[60, 120, 200, 400])
    ]

    if FLAGS.training:
        # Start training
        joint.train(dataset,
                    batch_size=FLAGS.batch_size,
                    track_models=FLAGS.checkpoints,
                    track_summaries=FLAGS.summaries,
                    steps=FLAGS.steps,
                    gpu_frac=FLAGS.gpu_frac,
                    metrics=metrics)
    else:
        # Evaluate test
        results = joint.evaluate(dataset,
                                 data_mode=DataMode.TEST,
                                 batch_size=FLAGS.batch_size,
                                 track_summaries=FLAGS.summaries,
                                 gpu_frac=FLAGS.gpu_frac,
                                 metrics=metrics)
        logger.info(results)
