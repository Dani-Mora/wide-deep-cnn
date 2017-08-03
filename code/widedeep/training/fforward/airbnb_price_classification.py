from widedeep.model.model_base import CNNModel, LinearModel, MLP
from protodata.image_ops import get_image_specs
from widedeep.model.joint_model import JointClassifier
from widedeep.ops.losses import Optimizers, CrossEntropy
import widedeep.ops.metrics as metrics
import widedeep.utils as utils

from protodata.datasets.airbnb import AirbnbSettings
from protodata.datasets import Datasets
from protodata.data_ops import TrainMode, DataMode
from protodata.utils import get_data_location
from protodata.quantize import Quantize

import tensorflow as tf

logger = utils.get_logger('data')

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
    10000,
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
    TrainMode.CNN,
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

# CNN parameters
flags.DEFINE_bool(
    "data_augm",
    True,
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
    utils.NetworkModels.ALEXNET,
    "Network to use for the CNN"
)

flags.DEFINE_string(
    "cnn_weights",
    utils.NetworkWeights.PLACES_365,
    "Weights to use for the network"
)

# Linear parameters
flags.DEFINE_float(
    "linear_initial_lr",
    1e-1,
    "Initial learning rate for the linear model"
)

flags.DEFINE_integer(
    "linear_decay_steps",
    80000,
    "Steps at which learning rate decreases for the linear model")

flags.DEFINE_float(
    "linear_decay_rate",
    0.1,
    "Decrease rate of the learning rate for the linear model"
)

# MLP parameters
flags.DEFINE_float(
    "mlp_initial_lr",
    1e-1,
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


if __name__ == '__main__':

    # Select Airbnb dataset
    bins = [40, 80, 150, 300]
    dataset = AirbnbSettings(
        dataset_location=FLAGS.data_location,
        image_specs=get_image_specs(FLAGS.cnn_network,
                                    batch_size=FLAGS.batch_size,
                                    random_crop=FLAGS.data_augm),
        embedding_dimensions=FLAGS.embeddings,
        quantizer=Quantize(edges=bins, batch_size=FLAGS.batch_size)
    )

    # Define columns
    wide = dataset.get_wide_columns()
    deep = dataset.get_deep_columns()
    image = dataset.get_image_column()

    # Show columns
    logger.info('Using wide columns: {}'.format([x.name for x in wide]))
    logger.info('Using deep columns: {}'.format([x.name for x in deep]))
    logger.info('Using image columns: {}'.format([x.name for x in image]))

    # Build CNN for image analysis
    if FLAGS.cnn_weights is not None:
        cnn_weights = utils.get_model_weights(FLAGS.cnn_network,
                                              FLAGS.cnn_weights)
    else:
        cnn_weights = None

    cnn_model = CNNModel(
        'cnn',
        columns_list=image,
        cnn_layers=utils.get_network_definition(FLAGS.cnn_network),
        optimizer=Optimizers.RMS,
        initial_lr=FLAGS.cnn_initial_lr,
        decay_steps=FLAGS.cnn_decay_steps,
        decay_rate=FLAGS.cnn_decay_rate,
        weights=cnn_weights,
        excluded_layers=['fc8', 'fc7', 'fc6']
    )

    # Build linear model for memorization
    linear_model = LinearModel('linear',
                               columns_list=wide,
                               optimizer=Optimizers.Ftrl,
                               initial_lr=FLAGS.linear_initial_lr,
                               decay_steps=FLAGS.linear_decay_steps,
                               decay_rate=FLAGS.linear_decay_rate)

    # Build MLP for generalization
    mlp_model = MLP('mlp',
                    columns_list=deep,
                    layers=utils.get_network_definition(FLAGS.mlp_network),
                    optimizer=Optimizers.Momentum,
                    initial_lr=FLAGS.mlp_initial_lr,
                    decay_steps=FLAGS.mlp_decay_steps,
                    decay_rate=FLAGS.mlp_decay_rate,
                    momentum=0.9)

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

    # Define model, loss and metrics according to model type
    logger.info('Using classification problem for column {} with bins: {}'
                .format(dataset.target_class(), bins))
    model_metrics = [
        metrics.Accuracy(),
        metrics.AccuracyMode(dataset.get_num_classes()),
        metrics.AccuracyRandom(dataset.get_num_classes()),
        metrics.ConfusionMatrix(num_classes=dataset.get_num_classes())
    ]

    # Build model
    joint = JointClassifier(
        utils.get_default_output(),
        outputs=dataset.get_num_classes(),
        models=models,
        l1_reg=FLAGS.l1_regularization,
        l2_reg=FLAGS.l2_regularization,
        loss_fn=CrossEntropy(),
        clip_gradient=FLAGS.gradient_clip
    )

    if FLAGS.training:

        if FLAGS.validate:
            # Train using validation
            joint.train_and_validate(
                dataset=dataset,
                batch_size=FLAGS.batch_size,
                validate_interval=FLAGS.validate_interval,
                validate_steps=FLAGS.validate_steps,
                track_summaries=FLAGS.summaries,
                steps=FLAGS.steps,
                gpu_frac=FLAGS.gpu_frac,
                metrics=model_metrics,
                patience=FLAGS.patience,
                workers=FLAGS.workers,
                mem_dequeing=FLAGS.mem_deq
            )

        else:
            # Train without validation
            joint.train(dataset,
                        FLAGS.batch_size,
                        track_models=FLAGS.checkpoints,
                        track_summaries=FLAGS.summaries,
                        steps=FLAGS.steps,
                        gpu_frac=FLAGS.gpu_frac,
                        metrics=metrics,
                        workers=FLAGS.workers,
                        mem_dequeing=FLAGS.mem_deq)
    else:
        # Evaluate validation
        results = joint.evaluate(dataset,
                                 batch_size=FLAGS.batch_size,
                                 data_mode=DataMode.TEST,
                                 track_summaries=FLAGS.summaries,
                                 gpu_frac=FLAGS.gpu_frac,
                                 metrics=metrics,
                                 workers=FLAGS.workers,
                                 mem_dequeing=FLAGS.mem_deq)
        logger.info(results)
