""" Testing network definition functionalities """

from widedeep.utils import get_network_definition, NetworkModels
import widedeep.model.network_conf as nc

import tempfile
import os
import unittest


class NetworkConfTestCase(unittest.TestCase):

    def setUp(self):
        # Create two temporary files
        self.fd, self.path = tempfile.mkstemp(suffix='.conf')

    def tearDown(self):
        os.close(self.fd)
        os.remove(self.path)


class DynamicCNNConfTestCase(NetworkConfTestCase):

    def runTest(self):
        """ Create a CNN, write it into a file. Read it and write it again
        and check they match """
        c1 = nc.ConvLayer('first',
                          ksize=3,
                          stride=2,
                          group=1,
                          maps=25,
                          padding='valid',
                          lrn=True,
                          alpha=0.5,
                          beta=0.1,
                          radius=1,
                          bias=1,
                          pool='max2d',
                          pool_size=2,
                          pool_stride=1,
                          pool_padding='valid',
                          activation='relu')

        c2 = nc.ConvLayer('second',
                          ksize=5,
                          stride=1,
                          group=2,
                          maps=50,
                          padding='same',
                          pool='avg2d',
                          pool_size=3,
                          pool_stride=2,
                          pool_padding='same',
                          activation='sigmoid')

        f1 = nc.FullyCLayer('f1',
                            hidden=1024,
                            activation='tanh',
                            dropout=0.25,
                            batch_norm=True)

        f2 = nc.FullyCLayer('f2',
                            hidden=500,
                            activation='sigmoid',
                            dropout=0.15,
                            batch_norm=True)

        # Write config
        conf_cnn = nc.CNNConfig([c1, c2], [f1, f2])
        conf_cnn.write(self.path)

        # Now read written config and write it again
        conf_cnn2 = nc.CNNConfig()
        conf_cnn2.read(self.path)

        self.assertTrue(conf_cnn == conf_cnn2)


class AlexnetConfTestCase(NetworkConfTestCase):

    def runTest(self):
        """ Read alexnet, write it and check they match """
        test_cnn = nc.CNNConfig()
        test_cnn.read(get_network_definition(NetworkModels.ALEXNET))
        test_cnn.write(self.path)

        test_cnn2 = nc.CNNConfig()
        test_cnn2.read(self.path)

        self.assertTrue(test_cnn == test_cnn2)


class DynamicMLPConfTestCase(NetworkConfTestCase):

    def testRun(self):
        """ Create MLP, write it and then read and write it again and compare
        whether they match """

        f1 = nc.FullyCLayer(name='fc1',
                            hidden=10,
                            activation='relu',
                            dropout=0.0,
                            batch_norm=False)

        f2 = nc.FullyCLayer(name='fc2',
                            hidden=25,
                            activation='identity',
                            dropout=0.10,
                            batch_norm=True)

        f3 = nc.FullyCLayer(name='fc3',
                            hidden=100,
                            activation='sigmoid',
                            dropout=0.50,
                            batch_norm=False)

        f4 = nc.FullyCLayer(name='fc4',
                            hidden=1,
                            ctivation='tanh',
                            dropout=0.0,
                            batch_norm=True)

        # Create from scratch - MLP
        conf_mlp = nc.MLPConfig([f1, f2, f3, f4])
        conf_mlp.write(self.path)

        # Read and write into separate file
        conf_mlp2 = nc.MLPConfig()
        conf_mlp2.read(self.path)

        # Match files
        self.assertTrue(conf_mlp == conf_mlp2)


class SimpleMLPConfTestCase(NetworkConfTestCase):

    def runTest(self):
        """ Read handwritten network, write it and check they match """
        test_mlp = nc.MLPConfig()
        test_mlp.read(get_network_definition(NetworkModels.MLP))
        test_mlp.write(self.path)

        test_mlp2 = nc.MLPConfig()
        test_mlp2.read(self.path)

        self.assertTrue(test_mlp == test_mlp2)


if __name__ == '__main__':
    unittest.main()
