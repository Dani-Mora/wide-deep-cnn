# wide-deep-cnn
Framework extending the wide and deep approach for CNN networks.

Based on the [work from Heng-Tze et al](https://arxiv.org/abs/1606.07792) we developed a generalization of the wide and deep framework to multiple models ncluding Alexnet convolutional models. Both Imagenet and Places365 weights have been provided for warm start. This framework is intended to be used on general prupose problems apart from recommender system where wide and deep systems emerged.

# Instructions

First, load the submodules contain in the project by:

`git submodule init --recursive`

This repository has been coded to work with Python 3.x. Version 2.7 is not guaranteed to work. All libraries needed have been collected into the requirements file and can be installed with:

`pip install -r requirements`

We highly recommend the use of virtual environments for Python.

The framework has been developed using Tensorflow and the requirements install a gpu-ready version of it. In order to exploit the potential of your GPU you must follow [these steps](https://www.tensorflow.org/install/install_linux#nvidia_requirements_to_run_tensorflow_with_gpu_support) before running anything.


# Data

The datasets used in the code come from my other project [protodata](https://github.com/DaniUPC/protodata). In order to use other datasets please read the instructions in the link provided.

# Mixture Density Networks

Mixture Density Networks ([Bishop et al](http://publications.aston.ac.uk/373/)) has been very successful in several regression scenarios. We have integrated these networks into our Wide-Deep-CNN framework using Gaussian Mixtures.

# Examples

Usage examples of the functionalities of this framework can be found in [this folder](https://github.com/DaniUPC/wide-deep-cnn/tree/master/code/widedeep/training). The 'fforward' folder contains classification and regression examples for several datasets while the 'mixtures' folder contains regression examples of 3 datasets using Mixture Density Networks.

# Visualization

Tensorflow is built-in with a very powerful visualization tool called Tensorboard. By default, this framework will output a summary of the training/testing processes into a 'logs' folder in the root of the repository. To visualize its content, we must call:

`tensorboard --logdir=logs`

Once Tensorboard has been launched, we can use the visualization environmnet by browsing:

`http://localhost:6006`

# Future work

- Integration of random search for hyperparameter tuning.
- Fix multiple identical lines in logs
- Best model in validation has to be manually corrected in the 'checkpoint' in the logs. Correction must be done.
