# ! /usr/bin/python

import time
import os
import logging.config
import logging
import sys

from protodata.utils import download_file


class NetworkModels:
    ALEXNET = 'alexnet'
    VGG = 'vgg'
    MNIST = 'mnist'
    MLP = 'mlp'
    OWN = 'own'


class NetworkWeights:
    PLACES_365 = 'places_365'
    IMAGENET = 'imagenet'


CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(os.path.join(CODE_ROOT, '..'), '..')
DATA_ROOT = os.path.join(ROOT_DIR, 'data')

# Places365 models paths
MODELS_FOLDER = os.path.join(DATA_ROOT, 'models')

# Repository of external weight storage
WEIGHTS_REPO = 'https://bitbucket.org/Dani_UPC/wide-and-deep-resources/raw/master'  # noqa

# Default model output
MODEL_FOLDER = os.path.join(ROOT_DIR, 'logs')

LOG_CONFIG = os.path.join(os.path.join(CODE_ROOT, 'logs'), 'logging.conf')
LOG_OUT = os.path.join(os.path.join(CODE_ROOT, 'logs'), 'wide_deep.log')


def get_logger(name, level=logging.DEBUG, channel=sys.stdout):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Set handler level and set output to go into std output
    handler = logging.StreamHandler(channel)
    handler.setLevel(level)

    # Set specific format
    src_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(src_format)
    handler.setFormatter(formatter)

    # Add handler to main logger
    logger.addHandler(handler)
    return logger


logger = get_logger(__name__)


def get_model_weights(network, weights):
    """ Returns the path to the weights for the given model. If they do not exist,
    they are downloaded from the corresponding repository """
    weight_path = os.path.join(os.path.join(MODELS_FOLDER, network),
                               weights + '.npy')
    if not os.path.isfile(weight_path):
        logger.info("Weights not found. Downloading them ...")
        download_weights(network, weights, weight_path)
    return weight_path


def download_weights(network, weights, path):
    """ Downloads the weights from the external repository given the
    weight identifier """
    url = '/'.join([WEIGHTS_REPO, network, weights + '.npy'])
    logger.info("Downloading weights {} from {} into {}"
                .format(weights, url, path))
    download_file(url, path)


def get_network_definition(network_tag):
    return os.path.join(os.path.join(MODELS_FOLDER, network_tag),
                        network_tag + '.config')


def get_default_output():
    return MODEL_FOLDER


def current_time_id():
    """ Returns the current time ID in milliseconds """
    return int(round(time.time() * 1000))


def get_parent(path):
    """ Returns the parent folder of the given path """
    return os.path.abspath(os.path.join(path, os.pardir))


def get_filename_url(url):
    """
    Args:
        url: Input url
    Returns:
        Returns the corresponding file name and its extension
    """
    name = os.path.basename(url)
    return os.path.splitext(name)


def has_method(obj, method):
    """ Returns whether input object contains the given method """
    method_op = getattr(obj, method, None)
    return callable(method_op)


def get_parent_folder(folder):
    return os.path.abspath(os.path.join(folder, os.pardir))


def process_categories(cat_path):
    """ Returns the mapping between the identifier of a category in
    Places365 and its corresponding name
    Args:
        cat_path: Path containing the information about the Places365
            categories
    """
    result = {}
    with open(cat_path) as f:
        lines = f.readlines()
    # Map each position to the corresponding category
    for l in lines:
        parts = l.split(' ')
        raw_cat, num_cat = parts[0].rstrip(), int(parts[1].rstrip())
        category = raw_cat[3:]  # Erase prefix related to first letter
        result[num_cat] = category
    return result


def get_places365_categories():
    return os.path.join(DATA_ROOT, 'places_365_categories.txt')


class ModelNotFound(ValueError):
    pass


class VariableNotFound(AttributeError):
    pass


class MultipleFound(AttributeError):
    pass


class TypeError(ValueError):
    pass


class ResourceNotFound(ValueError):
    pass
