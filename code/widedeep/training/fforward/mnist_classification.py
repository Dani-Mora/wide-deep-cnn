from widedeep.model.model_base import CNNModel, LinearModel, MLP

from widedeep.model.joint_model import JointClassifier
from widedeep.ops.losses import Optimizers, CrossEntropy
import widedeep.utils as ut
from widedeep.ops.metrics import Accuracy, AccuracyRandom

from protodata.image_ops import get_image_specs
from protodata.data_ops import TrainMode, DataMode
from protodata.datasets import Datasets
from protodata.datasets.mnist import MnistSettings
from protodata.utils import get_data_location

import tensorflow as tf

logger = ut.get_logger('data')

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    "data_location",
    get_data_location(Datasets.MNIST),
    "Where data is stored"
)

flags.DEFINE_integer(
    "batch_size",
    16,
    "Batch size to use."
)

flags.DEFINE_integer(
    "validate_steps",
    32,
    "Batch size to use for validation"
)

flags.DEFINE_integer(
    "validate_interval",
    500,
    "Batch size to use for validation"
)

flags.DEFINE_string(
    "network",
    ut.NetworkModels.MNIST,
    "Network to use"
)

flags.DEFINE_integer(
    "summaries",
    50,
    "Steps between summaries."
)

flags.DEFINE_integer(
    "checkpoints",
    500,
    "Steps between model checkpoints."
)

flags.DEFINE_integer("steps", 1500, "Steps to train.")

flags.DEFINE_float("gpu_frac", 0.70, "Percentage of GPU memory to use.")
# High levels of gpu_frac can lead to
# Error: failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED
flags.DEFINE_string("mode", TrainMode.CNN, "Architecture to use")

flags.DEFINE_bool("training", True, "Execution mode")
flags.DEFINE_bool("validate", True, "Whether to use validation for training")
flags.DEFINE_integer(
    "patience",
    2,
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

# CNN parameters
flags.DEFINE_bool(
    "data_augm",
    True,
    "Whether to randomly crop images or not"
)

flags.DEFINE_float(
    "cnn_initial_lr",
    1e-2,
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
    "Decrease rate of the learning rate for the cnn model")

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

# Linear parameters
flags.DEFINE_float(
    "linear_initial_lr",
    1e-1,   "Initial learning rate for the linear model"
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
flags.DEFINE_string(
    "mlp_network",
    ut.NetworkModels.MLP,
    "Network to use for the MLP"
)


if __name__ == '__main__':

    # Select Airbnb dataset
    dataset = MnistSettings(
        dataset_location=FLAGS.data_location,
        image_specs=get_image_specs(FLAGS.network,
                                    batch_size=FLAGS.batch_size,
                                    mean=[0.0, 0.0, 0.0],
                                    random_crop=False))

    # Define columns
    wide = dataset.get_wide_columns(excluded_cols=['colorspace', 'label'])
    deep = dataset.get_deep_columns(excluded_cols=['colorspace', 'label'])
    image = dataset.get_image_column()

    # Build CNN with MNIST images models independently
    cnn_model = CNNModel('cnn',
                         columns_list=image,
                         cnn_layers=ut.get_network_definition(FLAGS.network),
                         optimizer=Optimizers.Momentum,
                         initial_lr=FLAGS.cnn_initial_lr,
                         decay_steps=FLAGS.cnn_decay_steps,
                         decay_rate=FLAGS.cnn_decay_rate,
                         weights=None,
                         momentum=0.9)

    # Build linear model with categorical pixel values
    linear_model = LinearModel('linear',
                               columns_list=wide,
                               optimizer=Optimizers.Ftrl,
                               initial_lr=FLAGS.linear_initial_lr,
                               decay_steps=FLAGS.linear_decay_steps,
                               decay_rate=FLAGS.linear_decay_rate)

    # Build MLP with pixel values
    mlp_model = MLP('mlp',
                    columns_list=deep,
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
    elif FLAGS.mode == TrainMode.CNN:
        models = [cnn_model]
    elif FLAGS.mode == TrainMode.ALL:
        models = [cnn_model, linear_model, mlp_model]

    # Create model
    joint = JointClassifier(ut.get_default_output(),
                            models=models,
                            loss_fn=CrossEntropy(),
                            l1_reg=FLAGS.l1_regularization,
                            l2_reg=FLAGS.l2_regularization,
                            outputs=dataset.get_num_classes(),
                            clip_gradient=FLAGS.gradient_clip)

    model_metrics = [
        Accuracy(),
        AccuracyRandom(num_classes=dataset.get_num_classes())
    ]

    if FLAGS.training:
        if FLAGS.validate:
            # Start training and validation
            joint.train_and_validate(dataset=dataset,
                                     batch_size=FLAGS.batch_size,
                                     validate_interval=FLAGS.validate_interval,
                                     validate_steps=FLAGS.validate_steps,
                                     track_summaries=FLAGS.summaries,
                                     steps=FLAGS.steps,
                                     patience=FLAGS.patience,
                                     gpu_frac=FLAGS.gpu_frac,
                                     metrics=model_metrics)
        else:
            # Start training
            joint.train(dataset=dataset,
                        batch_size=FLAGS.batch_size,
                        track_models=FLAGS.checkpoints,
                        track_summaries=FLAGS.summaries,
                        steps=FLAGS.steps,
                        gpu_frac=FLAGS.gpu_frac,
                        metrics=model_metrics)

    else:
        # Evaluate on test
        results = joint.evaluate(dataset,
                                 data_mode=DataMode.TEST,
                                 batch_size=FLAGS.batch_size,
                                 track_summaries=FLAGS.summaries,
                                 gpu_frac=FLAGS.gpu_frac,
                                 metrics=model_metrics)

        logger.info(results)
