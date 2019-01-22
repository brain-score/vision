import functools
import numpy as np
import os

from model_tools.activations import ActivationsExtractor
from model_tools.activations.tensorflow_slim import TensorflowSlimContainer


def unique_preserved_order(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


class TestWrapperMatch:
    def test_pytorch(self):
        from torch import nn

        class MyModel(nn.Module):
            pass

        model = MyModel()
        a = ActivationsExtractor(model)
        from model_tools.activations.pytorch import PytorchWrapper
        assert isinstance(a.get_activations, PytorchWrapper)

    def test_keras(self):
        from keras.applications.vgg19 import VGG19
        model = VGG19()
        a = ActivationsExtractor(model)
        from model_tools.activations.keras import KerasWrapper
        assert isinstance(a.get_activations, KerasWrapper)

    def test_tensorflowslim(self):
        from nets import nets_factory
        model_ctr = nets_factory.get_network_fn('inception_v1', num_classes=1001, is_training=False)
        model = TensorflowSlimContainer(model_ctr, batch_size=64, image_size=224)

        a = ActivationsExtractor(model)
        from model_tools.activations.tensorflow_slim import TensorflowSlimWrapper
        assert isinstance(a.get_activations, TensorflowSlimWrapper)


class TestNewModel:
    def test_pytorch(self):
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

        activations_extractor = ActivationsExtractor(
            MyModel(), preprocessing=functools.partial(load_preprocess_images, image_size=224), pca_components=None)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')]
        activations = activations_extractor(stimuli_paths=stimuli_paths, layers=['linear', 'relu2'])
        assert activations is not None
        assert len(activations['stimulus_path']) == 1
        assert len(activations['neuroid']) == 1000 + 1000

    def test_tensorflow_slim(self):
        import tensorflow as tf
        from preprocessing import vgg_preprocessing
        slim = tf.contrib.slim

        class MyModelWrapper(TensorflowSlimModel):
            def _create_inputs(self, batch_size, image_size):
                inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3])
                preprocess_image = vgg_preprocessing.preprocess_image
                return tf.map_fn(lambda image: preprocess_image(tf.image.convert_image_dtype(image, dtype=tf.uint8),
                                                                image_size, image_size), inputs)

            def _create_model(self, inputs):
                with tf.variable_scope('my_model', values=[inputs]) as sc:
                    end_points_collection = sc.original_name_scope + '_end_points'
                    # Collect outputs for conv2d, fully_connected and max_pool2d.
                    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                        outputs_collections=[end_points_collection]):
                        net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
                        net = slim.max_pool2d(net, [5, 5], 5, scope='pool1')
                        net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
                        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                        return net, end_points

            def _restore(self, weights):
                assert weights is None
                init = tf.initialize_all_variables()
                self._sess.run(init)

        activations = model_activations(model=MyModelWrapper, model_identifier='test_tensorflow_slim',
                                        layers=['my_model/pool2'],
                                        weights=None, pca_components=None)
        assert activations is not None
        assert len(activations['neuroid']) == 4 * 4 * 64
