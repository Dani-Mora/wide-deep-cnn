#!/usr/bin/python

import abc
import configparser
from configparser import NoOptionError
import os
from widedeep.utils import NetworkModels


class Layer(object):
    """ Generic class for a network layer. Field:
        - name: Identifier of the layer
        - activation: Activation function. Accepted:
            'relu', 'sigmoid', 'identify', 'tanh'. Default is 'relu'.
     """

    __metaclass__ = abc.ABCMeta

    def __init__(self, name=None, activation='relu', batch_norm=False):
        self.name = name
        if activation and activation not \
                in ['relu', 'identity', 'sigmoid', 'tanh']:
            raise ValueError('Pooling possible values: avg, max')
        self.activation = activation
        self.batch_norm = batch_norm

    def _check_activation(self):
        if self.activation not in ['relu', 'identity', 'sigmoid', 'tanh']:
            raise ValueError('Possible activation functions: identity, ' +
                             'relu and sigmoid')

    def from_config(self, name, config):
        section = config[name]
        self.name = section.get('name')
        try:
            func = section.get('activation')
            self.activation = func
        except NoOptionError:
            self.activation = 'relu'

        if config.has_option(name, 'batch_norm'):
            self.batch_norm = section.getboolean('batch_norm')

        self._from_section(section)
        self._check_activation()

    def store(self, name, config):
        config[name] = {}
        config[name]['name'] = self.name
        config[name]['batch_norm'] = str(self.batch_norm)
        self._store(config[name])
        config[name]['activation'] = str(self.activation)
        self._check_activation()

    @abc.abstractmethod
    def _from_section(self, section):
        """ Reads layer specific data from the configuration section """

    @abc.abstractmethod
    def _store(self, config):
        """ Stores layer-specific data into the cofiguration """

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class ConvLayer(Layer):
    """ Convolutional layer. Contains following information:
        - name: Identifier of the layer.
        - ksize: Size of the convolutional filter.
        - stride: Stride of the convolution.
        - group: Groups to use for convolution. Used only for Alexnet.
        - maps: Number of filters in the layer.
        - padding: Padding to apply in the convolution.
            Valid values: 'same', 'valid'. Default is 'same'.
        - lrn: Whether to apply Local Response Normalization (LRN).
            Default is False.
        - alpha: Alpha parameter of the LRN. Unused if lrn is False.
        - beta: Beta parameter of the LRN. Unused if lrn is False.
        - radius: Radius parameter of the LRN. Unused if lrn is False.
        - bias: Bias parameter of the LRN. Unused if lrn is False.
        - pool: Type of pooling to apply after the convolution.
            Accepted poolings: 'max', 'avg'. Set to None to disable.
        - pool_size: Size of the pooling. Unused if no pooling.
        - pool_stride: Stride of the pooling. Unused if no pooling.
        - pool_padding: Padding to apply in the pooling operation.
            Valid values: 'same', 'valid'. Default is 'same'.
        - activation: Activation function. Accepted:
            'relu', 'sigmoid', 'identify'. Default is 'relu'.
        - batch_norm: Whether to apply batch normalization before
            computing non-linearity
    """

    def __init__(self, name=None, ksize=None, stride=None, group=1, maps=None,
                 padding='same', lrn=False, alpha=None, beta=None,
                 radius=None, bias=None, pool=None, pool_size=None,
                 pool_stride=None, pool_padding='same', activation='relu',
                 batch_norm=False):
        super(ConvLayer, self).__init__(name, activation, batch_norm)
        self.ksize = ksize
        self.stride = stride
        self.group = group
        self.maps = maps
        self.padding = padding
        self.lrn = lrn
        self.alpha = alpha
        self.beta = beta
        self.radius = radius
        self.bias = bias
        self.pool = pool
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def _check_pool(self):
        if self.pool and self.pool not in ['avg2d', 'max2d']:
            raise ValueError('Pooling possible values: avg2d, max2d')

    def _check_padding(self):
        if self.padding not in ['same', 'valid']:
            raise ValueError('Invalid padding type %s' % self.padding)
        if self.pool_padding and self.padding not in ['same', 'valid']:
            raise ValueError('Invalid padding type %s' % self.padding)

    def _from_section(self, section):
        """ See base function """
        self.ksize = section.getint('kernel_size')
        self.stride = section.getint('kernel_stride')

        group = section.getint('group')
        self.group = group if group is not None else 1  # Group is 1 by default

        self.maps = section.getint('maps')
        self.padding = section.get('padding')

        self.lrn = section.getboolean('lrn')
        if self.lrn:
            self.alpha = section.getfloat('lrn_alpha')
            self.beta = section.getfloat('lrn_beta')
            self.radius = section.getfloat('lrn_radius')
            self.bias = section.getfloat('lrn_bias')

        self.pool = section.get('pooling_type')

        if self.pool:
            self.pool_size = section.getint('pool_size')
            self.pool_stride = section.getint('pool_stride')
            self.pool_padding = section.get('pool_padding')

        self._check_pool()
        self._check_padding()

    def _store(self, config):
        """ See base function """
        config['kernel_size'] = str(self.ksize)
        config['kernel_stride'] = str(self.stride)

        if self.group != 1:
            config['group'] = str(self.group)  # Save group only if not default

        config['maps'] = str(self.maps)
        config['padding'] = str(self.padding)

        config['lrn'] = str(self.lrn)
        if self.lrn:
            config['lrn_alpha'] = str(self.alpha)
            config['lrn_beta'] = str(self.beta)
            config['lrn_radius'] = str(self.radius)
            config['lrn_bias'] = str(self.bias)

        if self.pool:
            config['pooling_type'] = str(self.pool)
            config['pool_size'] = str(self.pool_size)
            config['pool_stride'] = str(self.pool_stride)
            config['pool_padding'] = str(self.pool_padding)

        self._check_pool()
        self._check_padding()


class FullyCLayer(Layer):
    """ Fully connected layer. Contains following information:
        - name: Identifier of the layer
        - hidden: Number of hidden units in the layer
        - activation: Non-linear function. Valid values:
            'relu', 'sigmoid', 'identity'. Default value is 'relu'
        - batch_norm: Whether to apply batch normalization before
            computing non-linearity
    """

    def __init__(self, name=None, hidden=None, activation='relu',
                 dropout=None, batch_norm=False):
        super(FullyCLayer, self).__init__(name, activation, batch_norm)
        self.hidden = hidden
        self.dropout = dropout

    def _check_dropout(self):
        if self.dropout and (self.dropout < 0.0 or self.dropout > 1.0):
            raise ValueError('Dropout value must be in interval [0,1]')

    def _from_section(self, section):
        """ See base function """
        self.hidden = section.getint('hidden')
        self.dropout = section.getfloat('dropout')
        self._check_dropout()

    def _store(self, config):
        """ See base function """
        self._check_dropout()
        config['hidden'] = str(self.hidden)
        config['dropout'] = str(self.dropout) if self.dropout else str('0.0')


class NetworkConfig(object):
    """ Configuration that represents the structure of a network.
    Each layer is included in a section and has a prefix that identifies
    the layer type. The identifier of the layer must be unique and in
    increasing order"""

    __metaclass__ = abc.ABCMeta

    CONV_PREFIX = 'Conv'
    FULLC_PREFIX = 'FullyC'

    def read(self, path):
        """ Reads from a serialized configuration file"""
        config = configparser.ConfigParser()
        if not os.path.isfile(str(path)):
            raise IOError('File not found %s' % path)
        config.read(path)
        self._read(config)

    @abc.abstractmethod
    def _read(self, config):
        """ Reads from a serialized configuration"""

    def write(self, path):
        """ Output the configuration into the given path """
        config = configparser.ConfigParser()
        self._write(config)
        with open(path, 'w') as f:
            config.write(f)

    @abc.abstractmethod
    def _write(self, config):
        """ Writes the desired content into the configuration file """

    @staticmethod
    def get_layers(config, prefix):
        layers = [i for i in config.sections() if str(i).startswith(prefix)]
        sorted(layers)
        return layers

    def read_layer(self, layer, config, layer_type):
        """ Reads a layer from the configuration given its type tag"""
        if layer_type == self.CONV_PREFIX:
            instance = ConvLayer()
        elif layer_type == self.FULLC_PREFIX:
            instance = FullyCLayer()
        else:
            raise ValueError('Unknown layer prefix %s' % layer_type)
        instance.from_config(layer, config)
        return instance

    def write_layer(self, layer, ident, config):
        """ Writes the layer with given identifier into the configuration, creating
         one section for the layer """
        if isinstance(layer, ConvLayer):
            prefix = self.CONV_PREFIX
        elif isinstance(layer, FullyCLayer):
            prefix = self.FULLC_PREFIX
        else:
            raise ValueError('Unknown layer {}' % layer)
        layer.store(prefix + str(ident), config)

    def _read_conv_layers(self, config):
        """ Reads the convolutional layers in the configuration
        from the configuration """
        return self._read_layers(config, self.CONV_PREFIX)

    def _read_fullyc_layers(self, config):
        """ Reads the fully connected layers in the configuration
        from the configuration """
        return self._read_layers(config, self.FULLC_PREFIX)

    def _read_layers(self, config, layer_type):
        """ Read the layers of the given type (in increasing order)
        from the configuration """
        layers = self.get_layers(config, layer_type)
        return [self.read_layer(i, config, layer_type) for i in layers]

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class CNNConfig(NetworkConfig):
    """ Class that includes the layer configuration of a CNN network

        Layer parameters are declared in individual sections:
            [Conv1]
            [Conv2]
            ...
            [FullyC1]
            [FullyC2]

    """

    def __init__(self, conv_layers=[], full_layers=[]):
        self.conv_layers = conv_layers
        self.full_layers = full_layers

    def _read(self, config):
        self.conv_layers = self._read_conv_layers(config)
        self.full_layers = self._read_fullyc_layers(config)

    def _write(self, config):
        # Write each convolutional layer
        for (i, l) in enumerate(self.conv_layers):
            self.write_layer(l, i, config)

        # Write each fully connected layer afterwards
        for (i, l) in enumerate(self.full_layers):
            self.write_layer(l, i, config)

    def __str__(self):
        def str_layers(layers):
            return ','.join([str(l) for l in layers])
        return str_layers(self.conv_layers) + '|' + \
            str_layers(self.conv_layers)


class MLPConfig(NetworkConfig):
    """ Class that includes the layer configuration of a
        Multylayer Perceptron Network

        Layer parameters are declared in individual sections:
            [FullyC1]
            [FullyC2]

    """

    def __init__(self, hidden_layers=[]):
        self.hidden_layers = hidden_layers

    def _read(self, config):
        self.hidden_layers = self._read_fullyc_layers(config)

    def _write(self, config):
        # Write one section for each fully connected layer
        for (i, l) in enumerate(self.hidden_layers):
            self.write_layer(l, i, config)

    def __str__(self):
        def str_layers(layers):
            return ','.join([str(l) for l in layers])
        return str_layers(self.hidden_layers)


def create_mlp(layers, hiddens, dropout, activation='relu', batch_norm=False):
    """ Creates a MLP definition template using the input information """
    fcs = []
    for i in range(layers):
        fcs.append(FullyCLayer(name='fc' + str(i),
                               hidden=hiddens,
                               dropout=dropout,
                               activation=activation,
                               batch_norm=batch_norm))
    return MLPConfig(hidden_layers=fcs)


def create_cnn(net_type, dropout):
    """ Creates a cnn net configuration with the input dropout for the
    hidden layers """
    if NetworkModels.ALEXNET == net_type:
        return create_alexnet(dropout)
    elif NetworkModels.VGG == net_type:
        return create_vgg(dropout)
    elif NetworkModels.MNIST == net_type:
        return create_mnist(dropout)
    else:
        raise ValueError('Not valid network type %s' % net_type)


def create_alexnet(dropout):
    """ Creates an Alexnet configuration template given the input dropout
    for the hidden layers """
    # Convolutional layers
    conv_layers = [
        ConvLayer(name='conv1', ksize=11, stride=4, maps=96, padding='valid',
                  lrn=True, alpha=2e-05, beta=0.75, radius=2, bias=1.0,
                  pool='max2d', pool_size=3, pool_stride=2,
                  pool_padding='valid', activation='relu'),
        ConvLayer(name='conv2', ksize=5, stride=1, maps=256, group=2,
                  padding='same', lrn=True, alpha=2e-05, beta=0.75, radius=2,
                  bias=1.0, pool='max2d', pool_size=3, pool_stride=2,
                  pool_padding='valid', activation='relu'),
        ConvLayer(name='conv3', ksize=3, stride=1, maps=384, padding='same',
                  lrn=False, activation='relu'),
        ConvLayer(name='conv4', ksize=3, stride=1, group=2, maps=384,
                  padding='same', lrn=False, activation='relu'),
        ConvLayer(name='conv5', ksize=3, stride=1, group=2, maps=256,
                  padding='same', lrn=False, pool='max2d', pool_size=3,
                  pool_stride=2, pool_padding='valid', activation='relu')
    ]

    # Fully connected layers
    fully_layers = [
        FullyCLayer(name='fc6', hidden=4096,
                    dropout=dropout, activation='relu'),
        FullyCLayer(name='fc7', hidden=4096,
                    dropout=dropout, activation='relu')
    ]

    return CNNConfig(conv_layers=conv_layers, full_layers=fully_layers)


def create_vgg(dropout):
    """ Creates a VGG configuration template given the input dropout
    for the hidden layers """
    # Convolutional layers
    conv_layers = [
        ConvLayer(name='conv1_1', ksize=3, stride=1, maps=64,
                  padding='same', activation='relu'),
        ConvLayer(name='conv1_2', ksize=3, stride=1, maps=64, padding='same',
                  pool='max2d', pool_size=2, pool_stride=2,
                  pool_padding='same', activation='relu'),

        ConvLayer(name='conv2_1', ksize=3, stride=1, maps=128, padding='same',
                  activation='relu'),
        ConvLayer(name='conv2_2', ksize=3, stride=1, maps=128, padding='same',
                  pool='max2d', pool_size=2, pool_stride=2,
                  pool_padding='same', activation='relu'),

        ConvLayer(name='conv3_1', ksize=3, stride=1, maps=256, padding='same',
                  activation='relu'),
        ConvLayer(name='conv3_2', ksize=3, stride=1, maps=256, padding='same',
                  activation='relu'),
        ConvLayer(name='conv3_3', ksize=3, stride=1, maps=256, padding='same',
                  pool='max2d', pool_size=2, pool_stride=2,
                  pool_padding='same', activation='relu'),

        ConvLayer(name='conv4_1', ksize=3, stride=1, maps=512, padding='same',
                  activation='relu'),
        ConvLayer(name='conv4_2', ksize=3, stride=1, maps=512, padding='same',
                  activation='relu'),
        ConvLayer(name='conv4_3', ksize=3, stride=1, maps=512, padding='same',
                  pool='max2d', pool_size=2, pool_stride=2,
                  pool_padding='same', activation='relu'),

        ConvLayer(name='conv5_1', ksize=3, stride=1, maps=512, padding='same',
                  activation='relu'),
        ConvLayer(name='conv5_2', ksize=3, stride=1, maps=512, padding='same',
                  activation='relu'),
        ConvLayer(name='conv5_3', ksize=3, stride=1, maps=512, padding='same',
                  pool='max2d', pool_size=2, pool_stride=2,
                  pool_padding='same', activation='relu')
    ]

    # Fully connected layers
    fully_layers = [
        FullyCLayer(name='fc6', hidden=4096,
                    dropout=dropout, activation='relu'),
        FullyCLayer(name='fc7', hidden=4096,
                    dropout=dropout, activation='relu')
    ]

    return CNNConfig(conv_layers=conv_layers, full_layers=fully_layers)


def create_mnist(dropout):
    """ Creates a CNN small network for the MNIST dataset """
    # Convolutional layers
    conv_layers = [
        ConvLayer(name='conv1_1', ksize=7, stride=1, maps=64, padding='same',
                  lrn=True, alpha=2e-05, beta=0.75, radius=2, bias=1.0,
                  pool='max2d', pool_size=3, pool_stride=2,
                  pool_padding='valid', activation='relu'),
        ConvLayer(name='conv1_2', ksize=7, stride=1, maps=64, padding='same',
                  lrn=True, alpha=2e-05, beta=0.75, radius=2, bias=1.0,
                  pool='max2d', pool_size=3, pool_stride=2,
                  pool_padding='valid', activation='relu'),
    ]

    # Fully connected layers
    fully_layers = [
        FullyCLayer(name='fc6', hidden=1024,
                    dropout=dropout, activation='relu'),
        FullyCLayer(name='fc7', hidden=1024,
                    dropout=dropout, activation='relu')
    ]

    return CNNConfig(conv_layers=conv_layers, full_layers=fully_layers)
