import functools
import os

import numpy as np
import pytest
from brainio_base.stimuli import StimulusSet
from result_caching import cache

from model_tools.activations import KerasWrapper, PytorchWrapper, TensorflowWrapper


def unique_preserved_order(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


def pytorch_custom():
    import torch
    from torch import nn
    from model_tools.activations.pytorch import load_preprocess_images

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
            self.relu1 = torch.nn.ReLU()
            linear_input_size = np.power((224 - 3 + 2 * 0) / 1 + 1, 2) * 2
            self.linear = torch.nn.Linear(int(linear_input_size), 1000)
            self.relu2 = torch.nn.ReLU()  # can't get named ReLU output otherwise

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            x = self.relu2(x)
            return x

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    return functools.partial(PytorchWrapper, model=MyModel(), preprocessing=preprocessing), ['linear', 'relu2']


def pytorch_alexnet():
    from torchvision.models.alexnet import alexnet
    from model_tools.activations.pytorch import load_preprocess_images

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    return functools.partial(PytorchWrapper, model=alexnet(), preprocessing=preprocessing), \
           ['features.12', 'classifier.5']


def keras_vgg19():
    from keras.applications.vgg19 import VGG19, preprocess_input
    from model_tools.activations.keras import load_images
    preprocessing = lambda image_filepaths: preprocess_input(load_images(image_filepaths, image_size=224))
    return functools.partial(KerasWrapper, model=VGG19(), preprocessing=preprocessing), ['block3_pool']


@cache()  # cache to avoid re-initializing variable scope
def tfslim_custom():
    from model_tools.activations.tensorflow import load_resize_image
    import tensorflow as tf
    slim = tf.contrib.slim

    image_size = 224
    placeholder = tf.placeholder(dtype=tf.string, shape=[64])
    preprocess = lambda image_path: load_resize_image(image_path, image_size)
    preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)

    with tf.variable_scope('my_model', values=[preprocess]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(preprocess, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [5, 5], 5, scope='pool1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            endpoints = slim.utils.convert_collection_to_dict(end_points_collection)

    session = tf.Session()
    session.run(tf.initialize_all_variables())
    return functools.partial(TensorflowWrapper, identifier='tf-custom',
                             endpoints=endpoints, inputs=placeholder, session=session), ['my_model/pool2']


@cache()  # cache to avoid re-initializing variable scope
def tfslim_vgg16():
    import tensorflow as tf
    from nets import nets_factory
    from preprocessing import vgg_preprocessing
    from model_tools.activations.tensorflow import load_resize_image

    image_size = 224
    placeholder = tf.placeholder(dtype=tf.string, shape=[64])
    preprocess_image = lambda image: vgg_preprocessing.preprocess_image(
        image, image_size, image_size, resize_side_min=image_size)
    preprocess = lambda image_path: preprocess_image(load_resize_image(image_path, image_size))
    preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)

    model_ctr = nets_factory.get_network_fn('vgg_16', num_classes=1001, is_training=False)
    logits, endpoints = model_ctr(preprocess)

    session = tf.Session()
    session.run(tf.initialize_all_variables())
    return functools.partial(TensorflowWrapper, identifier='tf-vgg16',
                             endpoints=endpoints, inputs=placeholder, session=session), ['vgg_16/pool5']


@pytest.mark.parametrize("image_name", ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png'])
@pytest.mark.parametrize("pca_components", [None])
@pytest.mark.parametrize("provider", [
    pytorch_custom, pytorch_alexnet,
    keras_vgg19,
    tfslim_custom, tfslim_vgg16])
def test_from_image_path(provider, image_name, pca_components):
    stimuli_paths = [os.path.join(os.path.dirname(__file__), image_name)]

    extractor_ctr, layers = provider()
    activations_extractor = extractor_ctr(pca_components=None)
    activations = activations_extractor.from_paths(stimuli_paths=stimuli_paths, layers=layers)

    assert activations is not None
    assert len(activations['stimulus_path']) == 1
    assert len(np.unique(activations['layer'])) == len(layers)


@pytest.mark.parametrize("pca_components", [None])
@pytest.mark.parametrize("provider", [
    pytorch_custom, pytorch_alexnet,
    keras_vgg19,
    tfslim_custom, tfslim_vgg16])
def test_from_stimulus_set(provider, pca_components):
    image_names = ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png']
    stimulus_set = StimulusSet([{'image_id': image_name, 'some_meta': image_name[::-1]}
                                for image_name in image_names])
    stimulus_set.image_paths = {image_name: os.path.join(os.path.dirname(__file__), image_name)
                                for image_name in image_names}

    extractor_ctr, layers = provider()
    activations_extractor = extractor_ctr(pca_components=None)
    activations = activations_extractor.from_stimulus_set(stimulus_set, layers=layers)

    assert activations is not None
    assert set(activations['image_id'].values) == set(image_names)
    assert all(activations['some_meta'].values == [image_name[::-1] for image_name in image_names])
    assert len(np.unique(activations['layer'])) == len(layers)
