import functools
import numpy as np
import os
import pytest
import xarray as xr
from pathlib import Path

from brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.activations import KerasWrapper, PytorchWrapper, TensorflowSlimWrapper
from brainscore_vision.model_helpers.activations.core import flatten
from brainscore_vision.model_helpers.activations.pca import LayerPCA


def unique_preserved_order(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


def pytorch_custom():
    import torch
    from torch import nn
    from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

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
    return PytorchWrapper(model=MyModel(), preprocessing=preprocessing)


def pytorch_alexnet():
    from torchvision.models.alexnet import alexnet
    from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    return PytorchWrapper(model=alexnet(pretrained=True), preprocessing=preprocessing)


def pytorch_alexnet_resize():
    from torchvision.models.alexnet import alexnet
    from brainscore_vision.model_helpers.activations.pytorch import load_images, torchvision_preprocess
    from torchvision import transforms
    torchvision_preprocess_input = transforms.Compose([transforms.Resize(224), torchvision_preprocess()])

    def preprocessing(paths):
        images = load_images(paths)
        images = [torchvision_preprocess_input(image) for image in images]
        images = np.concatenate(images)
        return images

    return PytorchWrapper(alexnet(pretrained=True), preprocessing, identifier='alexnet-resize')


def pytorch_transformer_substitute():
    import torch
    from torch import nn
    from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

    class MyTransformer(nn.Module):
        def __init__(self):
            super(MyTransformer, self).__init__()
            self.conv = torch.nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3)
            self.relu1 = torch.nn.ReLU()
            linear_input_size = (224 ** 2 - 2) * 2
            self.linear = torch.nn.Linear(int(linear_input_size), 1000)
            self.relu2 = torch.nn.ReLU()  # logit out needs to be 1000

        def forward(self, x):
            x = x.view(*x.shape[:2], -1)
            x = self.conv(x)
            x = self.relu1(x)
            x = x.view(x.shape[0], -1)
            x = self.linear(x)
            x = self.relu2(x)

            return x

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    return PytorchWrapper(model=MyTransformer(), preprocessing=preprocessing)


def keras_vgg19():
    import keras
    from keras.applications.vgg19 import VGG19, preprocess_input
    from brainscore_vision.model_helpers.activations.keras import load_images
    keras.backend.clear_session()
    preprocessing = lambda image_filepaths: preprocess_input(load_images(image_filepaths, image_size=224))
    return KerasWrapper(model=VGG19(), preprocessing=preprocessing)


def tfslim_custom():
    from brainscore_vision.model_helpers.activations.tensorflow import load_resize_image
    import tensorflow as tf
    slim = tf.contrib.slim
    tf.compat.v1.reset_default_graph()

    image_size = 224
    placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[64])
    preprocess = lambda image_path: load_resize_image(image_path, image_size)
    preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)

    with tf.compat.v1.variable_scope('my_model', values=[preprocess]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(preprocess, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [5, 5], 5, scope='pool1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, 1000, scope='logits')
            endpoints = slim.utils.convert_collection_to_dict(end_points_collection)

    session = tf.compat.v1.Session()
    session.run(tf.compat.v1.initialize_all_variables())
    return TensorflowSlimWrapper(identifier='tf-custom', labels_offset=0,
                                 endpoints=endpoints, inputs=placeholder, session=session)


def tfslim_vgg16():
    import tensorflow as tf
    from nets import nets_factory
    from preprocessing import vgg_preprocessing
    from brainscore_vision.model_helpers.activations.tensorflow import load_resize_image
    tf.compat.v1.reset_default_graph()

    image_size = 224
    placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[64])
    preprocess_image = lambda image: vgg_preprocessing.preprocess_image(
        image, image_size, image_size, resize_side_min=image_size)
    preprocess = lambda image_path: preprocess_image(load_resize_image(image_path, image_size))
    preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)

    model_ctr = nets_factory.get_network_fn('vgg_16', num_classes=1001, is_training=False)
    logits, endpoints = model_ctr(preprocess)

    session = tf.compat.v1.Session()
    session.run(tf.compat.v1.initialize_all_variables())
    return TensorflowSlimWrapper(identifier='tf-vgg16', labels_offset=1,
                                 endpoints=endpoints, inputs=placeholder, session=session)


models_layers = [
    pytest.param(pytorch_custom, ['linear', 'relu2']),
    pytest.param(pytorch_alexnet, ['features.12', 'classifier.5'], marks=pytest.mark.memory_intense),
    pytest.param(pytorch_transformer_substitute, ['relu1']),
    pytest.param(keras_vgg19, ['block3_pool'], marks=pytest.mark.memory_intense),
    pytest.param(tfslim_custom, ['my_model/pool2'], marks=pytest.mark.memory_intense),
    pytest.param(tfslim_vgg16, ['vgg_16/pool5'], marks=pytest.mark.memory_intense),
]

# exact microsaccades for pytorch_alexnet, grayscale.png, for 1 and 10 number_of_trials
exact_microsaccades = {"x_degrees": {1: np.array([0.]),
                                     10: np.array([0., -0.00639121, -0.02114204, -0.02616418, -0.02128906,
                                                   -0.00941355, 0.00596172, 0.02166913, 0.03523793, 0.04498976])},
                       "y_degrees": {1: np.array([0.]),
                                     10: np.array([0., 0.0144621, 0.00728107, -0.00808922, -0.02338324, -0.0340791,
                                                   -0.03826824, -0.03578336, -0.02753704, -0.01503068])},
                       "x_pixels": {1: np.array([0.]),
                                    10: np.array([0., -0.17895397, -0.59197722, -0.73259714, -0.59609364, -0.26357934,
                                                  0.16692818, 0.60673569, 0.98666196, 1.25971335])},
                       "y_pixels": {1: np.array([0.]),
                                    10: np.array([0., 0.40493885, 0.20386999, -0.22649819, -0.65473077, -0.95421482,
                                                  -1.07151061, -1.00193403, -0.77103707, -0.42085896])}}


@pytest.mark.parametrize("image_name", ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png',
                                        'palletized.png'])
@pytest.mark.parametrize(["pca_components", "logits"], [(None, True), (None, False), (5, False)])
@pytest.mark.parametrize(["model_ctr", "layers"], models_layers)
def test_from_image_path(model_ctr, layers, image_name, pca_components, logits):
    stimuli_paths = [os.path.join(os.path.dirname(__file__), image_name)]

    activations_extractor = model_ctr()
    if pca_components:
        LayerPCA.hook(activations_extractor, pca_components)
    activations = activations_extractor.from_paths(stimuli_paths=stimuli_paths,
                                                   layers=layers if not logits else None)

    assert activations is not None
    assert len(activations['stimulus_path']) == 1
    assert len(np.unique(activations['layer'])) == len(layers) if not logits else 1
    if logits and not pca_components:
        assert len(activations['neuroid']) == 1000
    elif pca_components is not None:
        assert len(activations['neuroid']) == pca_components * len(layers)
    import gc
    gc.collect()  # free some memory, we're piling up a lot of activations at this point
    return activations


@pytest.mark.parametrize("image_name", ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png',
                                        'palletized.png'])
@pytest.mark.parametrize(["model_ctr", "layers"], models_layers)
@pytest.mark.parametrize("number_of_trials", [1, 5, 25])
def test_require_variance_has_shift_coords(model_ctr, layers, image_name, number_of_trials):
    stimulus_paths = [os.path.join(os.path.dirname(__file__), image_name)]
    activations_extractor = model_ctr()
    # when using microsaccades, the ModelCommitment sets its visual angle. Since this test skips the ModelCommitment,
    #  we set it here manually.
    activations_extractor._extractor.set_visual_degrees(8.)

    activations = activations_extractor(stimuli=stimulus_paths, layers=layers, number_of_trials=number_of_trials,
                                        require_variance=True)

    assert activations is not None
    assert len(activations['microsaccade_shift_x_pixels']) == number_of_trials * len(stimulus_paths)
    assert len(activations['microsaccade_shift_y_pixels']) == number_of_trials * len(stimulus_paths)
    assert len(activations['microsaccade_shift_x_degrees']) == number_of_trials * len(stimulus_paths)
    assert len(activations['microsaccade_shift_y_degrees']) == number_of_trials * len(stimulus_paths)


@pytest.mark.parametrize("image_name", ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png',
                                        'palletized.png'])
@pytest.mark.parametrize(["model_ctr", "layers"], models_layers)
@pytest.mark.parametrize("require_variance", [False, True])
@pytest.mark.parametrize("number_of_trials", [1, 2, 10])
def test_require_variance_presentation_length(model_ctr, layers, image_name, require_variance, number_of_trials):
    stimulus_paths = [os.path.join(os.path.dirname(__file__), image_name)]
    activations_extractor = model_ctr()
    # when using microsaccades, the ModelCommitment sets its visual angle. Since this test skips the ModelCommitment,
    #  we set it here manually.
    activations_extractor._extractor.set_visual_degrees(8.)

    activations = activations_extractor(stimuli=stimulus_paths, layers=layers,
                                        number_of_trials=number_of_trials, require_variance=require_variance)

    assert activations is not None
    if require_variance:
        assert len(activations['presentation']) == number_of_trials
    else:
        assert len(activations['presentation']) == 1


@pytest.mark.parametrize("image_name", ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png',
                                        'palletized.png'])
@pytest.mark.parametrize(["model_ctr", "layers"], models_layers)
def test_temporary_file_handling(model_ctr, layers, image_name):
    import tempfile
    stimulus_paths = [os.path.join(os.path.dirname(__file__), image_name)]
    activations_extractor = model_ctr()
    # when using microsaccades, the ModelCommitment sets its visual angle. Since this test skips the ModelCommitment,
    #  we set it here manually.
    activations_extractor._extractor.set_visual_degrees(8.)

    activations = activations_extractor(stimuli=stimulus_paths, layers=layers, number_of_trials=2,
                                        require_variance=True)
    temp_files = [f for f in os.listdir(tempfile.gettempdir()) if f.startswith('temp') and f.endswith('.png')]

    assert activations is not None
    assert len(temp_files) == 0


def _build_stimulus_set(image_names):
    stimulus_set = StimulusSet([{'stimulus_id': image_name, 'some_meta': image_name[::-1]}
                                for image_name in image_names])
    stimulus_set.stimulus_paths = {image_name: os.path.join(os.path.dirname(__file__), image_name)
                                   for image_name in image_names}
    return stimulus_set


@pytest.mark.parametrize("pca_components", [None, 5])
@pytest.mark.parametrize(["model_ctr", "layers"], models_layers)
def test_from_stimulus_set(model_ctr, layers, pca_components):
    image_names = ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png', 'palletized.png']
    stimulus_set = _build_stimulus_set(image_names)

    activations_extractor = model_ctr()
    if pca_components:
        LayerPCA.hook(activations_extractor, pca_components)
    activations = activations_extractor.from_stimulus_set(stimulus_set, layers=layers, stimuli_identifier=False)

    assert activations is not None
    assert set(activations['stimulus_id'].values) == set(image_names)
    assert all(activations['some_meta'].values == [image_name[::-1] for image_name in image_names])
    assert len(np.unique(activations['layer'])) == len(layers)
    if pca_components is not None:
        assert len(activations['neuroid']) == pca_components * len(layers)


@pytest.mark.memory_intense
@pytest.mark.parametrize("pca_components", [None, 1000])
def test_exact_activations(pca_components):
    activations = test_from_image_path(model_ctr=pytorch_alexnet_resize, layers=['features.12', 'classifier.5'],
                                       image_name='rgb.jpg', pca_components=pca_components, logits=False)
    path_to_expected = Path(__file__).parent / f'alexnet-rgb-{pca_components}.nc'
    expected = xr.load_dataarray(path_to_expected)

    # Originally, the `stimulus_path` Index was used to index into xarrays in Brain-Score, but this was changed
    #  as a part of PR #492 to a MultiIndex to allow metadata to be attached to multiple repetitions of the same
    #  `stimulus_path`. Old .nc files need to be updated to use the `presentation` index instead of `stimulus_path`,
    #  and instead of changing the extant activations, this test was simply modified to simulate that.
    expected = expected.rename({'stimulus_path': 'presentation'})

    assert (activations == expected).all()


@pytest.mark.memory_intense
@pytest.mark.parametrize("number_of_trials", [1, 10])
def test_exact_microsaccades(number_of_trials):
    image_name = 'grayscale.png'
    stimulus_paths = [os.path.join(os.path.dirname(__file__), image_name)]
    activations_extractor = pytorch_alexnet()
    # when using microsaccades, the ModelCommitment sets its visual angle. Since this test skips the ModelCommitment,
    #  we set it here manually.
    activations_extractor._extractor.set_visual_degrees(8.)
    # the exact microsaccades were computed at this extent
    assert activations_extractor._extractor._microsaccade_helper.microsaccade_extent_degrees == 0.05

    activations = activations_extractor(stimuli=stimulus_paths, layers=['features.12'],
                                        number_of_trials=number_of_trials, require_variance=True)

    assert activations is not None
    # test with np.isclose instead of == since while the arrays are visually equal, == often fails due to float errors
    assert np.isclose(activations['microsaccade_shift_x_degrees'].values,
                      exact_microsaccades['x_degrees'][number_of_trials],
                      rtol=1e-05,
                      atol=1e-08).all()
    assert np.isclose(activations['microsaccade_shift_y_degrees'].values,
                       exact_microsaccades['y_degrees'][number_of_trials],
                       rtol=1e-05,
                       atol=1e-08).all()
    assert np.isclose(activations['microsaccade_shift_x_pixels'].values,
                      exact_microsaccades['x_pixels'][number_of_trials],
                      rtol=1e-05,
                      atol=1e-08).all()
    assert np.isclose(activations['microsaccade_shift_y_pixels'].values,
                      exact_microsaccades['y_pixels'][number_of_trials],
                      rtol=1e-05,
                      atol=1e-08).all()


@pytest.mark.memory_intense
@pytest.mark.parametrize(["model_ctr", "internal_layers"], [
    (pytorch_alexnet, ['features.12', 'classifier.5']),
    (keras_vgg19, ['block3_pool']),
    (tfslim_vgg16, ['vgg_16/pool5']),
])
def test_mixed_layer_logits(model_ctr, internal_layers):
    stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')]

    activations_extractor = model_ctr()
    layers = internal_layers + ['logits']
    activations = activations_extractor(stimuli=stimuli_paths, layers=layers)
    assert len(np.unique(activations['layer'])) == len(internal_layers) + 1
    assert set(activations['layer'].values) == set(layers)
    assert unique_preserved_order(activations['layer'])[-1] == 'logits'


@pytest.mark.memory_intense
@pytest.mark.parametrize(["model_ctr", "expected_identifier"], [
    (pytorch_custom, 'MyModel'),
    (pytorch_alexnet, 'AlexNet'),
    (keras_vgg19, 'vgg19'),
])
def test_infer_identifier(model_ctr, expected_identifier):
    model = model_ctr()
    assert model.identifier == expected_identifier


def test_transformer_meta():
    model = pytorch_transformer_substitute()
    activations = model(stimuli=[os.path.join(os.path.dirname(__file__), 'rgb.jpg')], layers=['relu1'])
    assert hasattr(activations, 'channel')
    assert hasattr(activations, 'embedding')
    assert len(set(activations['neuroid_id'].values)) == len(activations['neuroid'])


def test_convolution_meta():
    model = pytorch_custom()
    activations = model(stimuli=[os.path.join(os.path.dirname(__file__), 'rgb.jpg')], layers=['conv1'])
    assert hasattr(activations, 'channel')
    assert hasattr(activations, 'channel_x')
    assert hasattr(activations, 'channel_y')
    assert len(set(activations['neuroid_id'].values)) == len(activations['neuroid'])


def test_conv_and_fc():
    model = pytorch_custom()
    activations = model(stimuli=[os.path.join(os.path.dirname(__file__), 'rgb.jpg')], layers=['conv1', 'linear'])
    assert set(activations['layer'].values) == {'conv1', 'linear'}


@pytest.mark.timeout(300)
def test_merge_large_layers():
    import torch
    from torch import nn
    from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

    class LargeModel(nn.Module):
        def __init__(self):
            super(LargeModel, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    model = PytorchWrapper(model=LargeModel(), preprocessing=preprocessing)
    activations = model(stimuli=[os.path.join(os.path.dirname(__file__), 'rgb.jpg')] * 64, layers=['conv', 'relu'])
    assert len(activations['neuroid']) == 394272
    assert len(set(activations['neuroid_id'].values)) == len(activations['neuroid'])
    assert set(activations['layer'].values) == {'conv', 'relu'}


class TestFlatten:
    def test_flattened_shape(self):
        A = np.random.rand(2560, 256, 6, 6)
        flattened = flatten(A)
        assert np.prod(flattened.shape) == np.prod(A.shape)
        assert flattened.shape[0] == A.shape[0]
        assert len(flattened.shape) == 2

    def test_indices_shape(self):
        A = np.random.rand(2560, 256, 6, 6)
        _, indices = flatten(A, return_index=True)
        assert len(indices.shape) == 2
        assert indices.shape[0] == np.prod(A.shape[1:])
        assert indices.shape[1] == 3  # for 256, 6, 6

    def test_match_flatten(self):
        A = np.random.rand(10, 256, 6, 6)
        flattened, indices = flatten(A, return_index=True)
        for layer in range(A.shape[0]):
            for i in range(np.prod(A.shape[1:])):
                value = flattened[layer][i]
                index = indices[i]
                assert A[layer][tuple(index)] == value

    def test_inverse(self):
        A = np.random.rand(2560, 256, 6, 6)
        flattened = flatten(A)
        A_ = np.reshape(flattened, [flattened.shape[0], 256, 6, 6])
        assert A.shape == A_.shape
        assert (A == A_).all()
