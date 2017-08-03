""" This test is intended to test whether the Places 365 network
weights can be loaded properly and output reasonable results """

from protodata.datasets.airbnb import AirbnbSettings
from protodata.data_ops import DataMode
from protodata.image_ops import get_image_specs
from protodata.datasets import Datasets
from protodata.utils import get_data_location

from widedeep.model.model_base import CNNModel

from widedeep.model.joint_model import JointClassifier
from widedeep.ops.losses import Optimizers, CrossEntropy
import widedeep.utils as ut

import tensorflow as tf

logger = ut.get_logger('data')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_location",
    get_data_location(Datasets.AIRBNB_PRICE),
    "Where data is stored"
)

flags.DEFINE_integer(
    "batch_size",
    64,
    "Batch size to use"
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
    50,
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

tf.app.flags.DEFINE_bool(
    'use_price',
    True,
    'Whether to predict price (True) or availability (False)'
)

flags.DEFINE_float(
    'classification',
    True,
    'Whether to build classification or regression dataset'
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

# CNN parameters
flags.DEFINE_bool(
    "data_augm",
    False,
    "Whether to randomly crop images or not"
)

flags.DEFINE_float(
    "cnn_initial_lr",
    1e-5,
    "Initial learning rate for the cnn model"
)

flags.DEFINE_integer(
    "cnn_decay_steps",
    100000,
    "Steps at which learning rate decreases for the cnn model"
)

flags.DEFINE_float(
    "cnn_decay_rate",
    0.1,
    "Decrease rate of the learning rate for the cnn model"
)

flags.DEFINE_string(
    "cnn_network",
    ut.NetworkModels.ALEXNET,
    "Network to use for the CNN"
)

flags.DEFINE_string(
    "cnn_weights",
    ut.NetworkWeights.PLACES_365,
    "Weights to use for the network"
)


if __name__ == '__main__':

    # Select Airbnb dataset
    dataset = AirbnbSettings(
        dataset_location=FLAGS.data_location,
        image_specs=get_image_specs(FLAGS.cnn_network,
                                    batch_size=FLAGS.batch_size,
                                    random_crop=FLAGS.data_augm),
        embedding_dimensions=16)  # Not used

    # Define columns
    image = dataset.get_image_column()
    logger.info('Using image columns: {}'.format([x.name for x in image]))

    # Build CNN for image analysis
    cnn_weights = ut.get_model_weights(FLAGS.cnn_network, FLAGS.cnn_weights) \
        if FLAGS.cnn_weights is not None else None
    models = [
        CNNModel('cnn',
                 columns_list=image,
                 cnn_layers=ut.get_network_definition(FLAGS.cnn_network),
                 optimizer=Optimizers.RMS,
                 initial_lr=FLAGS.cnn_initial_lr,
                 decay_steps=FLAGS.cnn_decay_steps,
                 decay_rate=FLAGS.cnn_decay_rate,
                 weights=cnn_weights,
                 excluded_layers=[])
    ]

    # Build model
    joint = JointClassifier(ut.get_default_output(),
                            outputs=365,
                            models=models,
                            l1_reg=FLAGS.l1_regularization,
                            l2_reg=FLAGS.l2_regularization,
                            loss_fn=CrossEntropy(),
                            clip_gradient=FLAGS.gradient_clip)

    # Train without validation
    joint.eval_places(dataset,
                      data_mode=DataMode.TRAINING,
                      steps=FLAGS.steps,
                      gpu_frac=FLAGS.gpu_frac,
                      workers=FLAGS.workers,
                      mem_dequeing=FLAGS.mem_deq)
