from model_tools.check_submission import check_models

import functools
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

"""
Template module for a brain model submission to brain-score
"""

def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ['resnet50_contrastive']


def get_model(model_identifier):
    """
    This method fetches an instance of a brain model. The instance has to implement the BrainModel interface in the
    brain-score project(see imports). To get a detailed explanation of how the interface hast to be implemented,
    check out the brain-score project(https://github.com/brain-score/brain-score), examples section :param name: the
    name of the model to fetch
    :return: the model instance, which implements the BrainModel interface
    """

    module = hub.KerasLayer("https://tfhub.dev/google/supcon/resnet_v1_50/imagenet/classification/1")

    model_preprocessing = keras.applications.resnet50.preprocess_input
    load_preprocess = lambda image_filepaths: model_preprocessing(load_images(image_filepaths, image_size=224))
    wrapper = KerasWrapper(model, load_preprocess)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    return ['ContrastiveModel/Encoder/BlockGroup0/BottleneckBlock_2/Conv2DFixedPadding_2/conv2d/kernel:0',
            'ContrastiveModel/Encoder/BlockGroup1/BottleneckBlock_3/Conv2DFixedPadding_2/conv2d/kernel:0',
            'ContrastiveModel/Encoder/BlockGroup2/BottleneckBlock_5/Conv2DFixedPadding_2/conv2d/kernel:0',
            'ContrastiveModel/Encoder/BlockGroup3/BottleneckBlock_2/Conv2DFixedPadding_2/conv2d/kernel:0',
            'ContrastiveModel/ProjectionHead/dense/kernel:0',
            'ContrastiveModel/ClassificationHead/dense/kernel:0']

def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BrainModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
