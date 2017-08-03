

from widedeep.model.model_base import LinearModel, MLP
from widedeep.model.joint_model import JointMDN
from widedeep.ops.distributions import GaussianMixture
from widedeep.ops.losses import Optimizers
import widedeep.utils as utils
from widedeep.ops import metrics

from protodata.datasets.scikit_dataset import DiabetesSettings
from protodata.datasets import Datasets
from protodata.data_ops import TrainMode, DataMode
from protodata.utils import get_data_location

import tensorflow as tf

logger = utils.get_logger('data')


flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    "data_location",
    get_data_location(Datasets.DIABETES),
    "Where data is stored"
)

flags.DEFINE_integer(
    "batch_size",
    64,
    "Batch size to use."
)

flags.DEFINE_integer(
    "validate_steps",
    100,
    "Batch size to use for validation"
)

flags.DEFINE_integer(
    "validate_interval",
    3000,
    "Batch size to use for validation"
)

flags.DEFINE_integer(
    "summaries",
    100,
    "Steps between summaries"
)

flags.DEFINE_integer(
    "checkpoints",
    5000000,
    "Steps between model checkpoints"
)


flags.DEFINE_integer(
    "steps",
    200000,
    "Steps to train"
)

flags.DEFINE_float(
    "gpu_frac",
    0.90,
    "Percentage of GPU memory to use"
)

flags.DEFINE_integer(
    "workers",
    2,
    "Workers to use for dequeing"
)

flags.DEFINE_integer(
    "mem_deq",
    2,
    "Memory factor (~GB) to use for dequeing"
)

flags.DEFINE_string(
    "mode",
    TrainMode.WIDE,
    "Architecture to use"
)

flags.DEFINE_bool(
    "training",
    True,
    "Execution mode"
)

flags.DEFINE_bool(
    "validate",
    True,
    "Whether to use validation for training"
)

flags.DEFINE_integer(
    "patience",
    3,
    "Consecutive allowed loss increases in early stopping"
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

# Gradient
flags.DEFINE_float(
    "gradient_clip",
    None,
    "If not None, value to use for clipping the gradient"
)

# Linear parameters
flags.DEFINE_float(
    "linear_initial_lr",
    1e-2,   "Initial learning rate for the linear model"
)

flags.DEFINE_integer(
    "linear_decay_steps",
    80000,
    "Steps at which learning rate decreases for the linear model"
)
flags.DEFINE_float(
    "linear_decay_rate",
    0.1,
    "Decrease rate of the learning rate for the linear model"
)

# MLP parameters
flags.DEFINE_float(
    "mlp_initial_lr",
    1e-2,
    "Initial learning rate for the MLP model"
)
flags.DEFINE_integer(
     "mlp_decay_steps",
     80000,
     "Steps at which learning rate decreases for the MLP model"
)

flags.DEFINE_float(
    "mlp_decay_rate",
    0.1,
    "Decrease rate of the learning rate for the MLP model"
)
flags.DEFINE_integer(
    "embeddings",
    8,
    "Number of dimensions for the embedding columns"
)
flags.DEFINE_string(
    "mlp_network",
    utils.NetworkModels.MLP,
    "Network to use for the MLP"
)

# Mixture density network parameters
flags.DEFINE_integer("K", 5, "Number of components in the mixture")


if __name__ == '__main__':

    # Select Airbnb dataset
    dataset = DiabetesSettings(dataset_location=FLAGS.data_location)

    # Define columns
    wide = dataset.get_wide_columns()
    logger.info('Using columns {}'.format(wide))

    # Create linear model
    linear_model = LinearModel('linear',
                               columns_list=wide,
                               optimizer=Optimizers.RMS,
                               initial_lr=FLAGS.linear_initial_lr,
                               decay_steps=FLAGS.linear_decay_steps,
                               decay_rate=FLAGS.linear_decay_rate)

    # Create deep model
    mlp_model = MLP('mlp',
                    columns_list=wide,
                    layers=utils.get_network_definition(FLAGS.mlp_network),
                    optimizer=Optimizers.RMS,
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
        raise ValueError('Unsupported option in Boston training %s'
                         % FLAGS.mode)

    model_metrics = [
        metrics.AbsError(),
        metrics.MeanPredError(),
        metrics.MedianPredError(),
        metrics.Mean(),
        metrics.Std(),
        metrics.AbsErrorInterval(cuts=[10, 15, 20]),
        metrics.RMSEInterval(cuts=[10, 15, 20])
    ]

    # Create model
    joint = JointMDN(mixture=GaussianMixture(components=FLAGS.K),
                     model_dir=utils.get_default_output(),
                     models=models,
                     outputs=dataset.get_num_classes(),
                     l1_reg=FLAGS.l1_regularization,
                     l2_reg=FLAGS.l2_regularization,
                     clip_gradient=FLAGS.gradient_clip)

    if FLAGS.training:
        # Start training
        joint.train(dataset,
                    batch_size=FLAGS.batch_size,
                    track_models=FLAGS.checkpoints,
                    track_summaries=FLAGS.summaries,
                    steps=FLAGS.steps,
                    gpu_frac=FLAGS.gpu_frac,
                    metrics=model_metrics,
                    log_steps=1)
    else:
        # Evaluate on test
        results = joint.evaluate(dataset,
                                 batch_size=FLAGS.batch_size,
                                 data_mode=DataMode.TRAINING,
                                 track_summaries=FLAGS.summaries,
                                 gpu_frac=FLAGS.gpu_frac,
                                 metrics=model_metrics)
        logger.info(results)
