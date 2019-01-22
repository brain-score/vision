import copy
import h5py
import inspect
import logging
import numpy as np
import os
from PIL import Image
from brainio_base.assemblies import NeuroidAssembly
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
from result_caching import store
from sklearn.decomposition import PCA
from tqdm import tqdm

from model_tools.activations.keras import KerasWrapper, preprocess as preprocess_keras
from model_tools.activations.pytorch import PytorchWrapper, preprocess_images as preprocess_pytorch
from model_tools.activations.tensorflow_slim import TensorflowSlimWrapper
from model_tools.utils import fullname, s3


class Defaults:
    batch_size = 64
    pca_components = 1000


class ActivationsExtractor:
    def __init__(self, model, preprocessing=None, pca_components=Defaults.pca_components,
                 batch_size=Defaults.batch_size):
        self._pca_components = pca_components
        self._batch_size = batch_size
        self._logger = logging.getLogger(fullname(self))
        self.model_name = model.__module__
        self.get_activations = self.infer_activations_method(model)
        self.preprocess = preprocessing or (lambda x: x)

    def infer_activations_method(self, model):
        wrappers = {
            'torch.nn.modules.module.Module': PytorchWrapper,
            'keras.engine.base_layer.Layer': KerasWrapper,
            'model_tools.activations.tensorflow_slim.TensorflowSlimContainer': TensorflowSlimWrapper,
        }
        function_name = 'get_activations'
        return self._infer(model, function_name=function_name, modulars=wrappers)

    def _infer(self, model, function_name, modulars):
        if hasattr(model, function_name):
            self._logger.debug("Using model's get_activations method")
            return model.get_activations
        # Instead of using `isinstance`, we manually go up the type hierarchy.
        # This is so that not everyone has to install all possible frameworks
        type_hierarchy = [_class.__module__ + "." + _class.__name__ for _class in inspect.getmro(type(model))]
        for type_name in type_hierarchy:
            if type_name in modulars:
                modular_ctr = modulars[type_name]
                modular = modular_ctr(model)
                return modular
        raise NotImplementedError(f"model does not define `{function_name}` "
                                  f"and no suitable wrapper found: {type_hierarchy} "
                                  f"(available: {list(modulars.keys())})")

    def __call__(self, stimuli_paths, layers):
        # PCA
        def get_activations(inputs, reduce_dimensionality):
            return self._get_activations_batched(inputs,
                                                 layers=layers, batch_size=self._batch_size,
                                                 reduce_dimensionality=reduce_dimensionality)

        reduce_dimensionality = self._initialize_dimensionality_reduction(
            model_name=self.model_name, pca_components=self._pca_components, get_image_activations=get_activations)
        # actual stimuli
        self._logger.info('Running stimuli')
        layer_activations = get_activations(stimuli_paths, reduce_dimensionality=reduce_dimensionality)

        self._logger.info('Packaging into assembly')
        return self._package(layer_activations, stimuli_paths)

    def _get_activations_batched(self, inputs, layers, batch_size, reduce_dimensionality):
        layer_activations = None
        for batch_start in tqdm(range(0, len(inputs), batch_size), unit_scale=batch_size, desc="activations"):
            batch_end = min(batch_start + batch_size, len(inputs))
            self._logger.debug('Batch %d->%d/%d', batch_start, batch_end, len(inputs))
            batch_inputs = inputs[batch_start:batch_end]
            batch_activations = self._get_batch_activations(batch_inputs, layer_names=layers, batch_size=batch_size)
            batch_activations = self._change_layer_activations(batch_activations, reduce_dimensionality,
                                                               keep_name=True, multithread=True)
            if layer_activations is None:
                layer_activations = copy.copy(batch_activations)
            else:
                for layer_name, layer_output in batch_activations.items():
                    layer_activations[layer_name] = np.concatenate((layer_activations[layer_name], layer_output))

        return layer_activations

    def _initialize_dimensionality_reduction(self, model_name, pca_components, get_image_activations):
        if pca_components is None:
            return flatten

        pca = self._compute_dimensionality_reduction(model_name=model_name, pca_components=pca_components,
                                                     get_image_activations=get_image_activations)

        # define dimensionality reduction method for external use
        def reduce_dimensionality(layer_name, layer_activations):
            layer_activations = flatten(layer_name, layer_activations)
            if layer_activations.shape[1] < pca_components:
                self._logger.debug(f"layer {layer_name} activations are smaller than pca components: "
                                   f"{layer_activations.shape} -- not performing PCA")
                return layer_activations
            return pca[layer_name].transform(layer_activations)

        return reduce_dimensionality

    @store(identifier_ignore=['get_image_activations'])
    def _compute_dimensionality_reduction(self, model_name, pca_components, get_image_activations):
        self._logger.info('Pre-computing principal components')  # TODO: ensure alignment
        self._logger.debug('Retrieving ImageNet activations')
        imagenet_paths = self._get_imagenet_val(pca_components)
        imagenet_activations = get_image_activations(imagenet_paths, reduce_dimensionality=flatten)
        self._logger.debug('Computing ImageNet principal components')
        progress = tqdm(total=len(imagenet_activations), desc="layer principal components")

        def compute_layer_pca(activations):
            if activations.shape[1] <= pca_components:
                self._logger.debug(f"Not computing principal components for activations {activations.shape} "
                                   f"as shape is small enough already")
                pca = None
            else:
                pca = PCA(n_components=pca_components)
                pca = pca.fit(activations)
            progress.update(1)
            return pca

        pca = self._change_layer_activations(imagenet_activations, compute_layer_pca, multithread=True)
        progress.close()
        return pca

    def _get_imagenet_val(self, num_images):
        num_classes = 1000
        num_images_per_class = (num_images - 1) // num_classes
        base_indices = np.arange(num_images_per_class).astype(int)
        indices = []
        for i in range(num_classes):
            indices.extend(50 * i + base_indices)
        for i in range((num_images - 1) % num_classes + 1):
            indices.extend(50 * i + np.array([num_images_per_class]).astype(int))

        framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
        imagenet_filepath = os.getenv('CM_IMAGENET_PATH', os.path.join(framework_home, 'imagenet2012.hdf5'))
        imagenet_dir = f"{imagenet_filepath}-files"
        os.makedirs(imagenet_dir, exist_ok=True)

        if not os.path.isfile(imagenet_filepath):
            os.makedirs(os.path.dirname(imagenet_filepath), exist_ok=True)
            self._logger.debug(f"Downloading ImageNet validation to {imagenet_filepath}")
            s3.download_file("imagenet2012-val.hdf5", imagenet_filepath)

        filepaths = []
        with h5py.File(imagenet_filepath, 'r') as f:
            for index in indices:
                imagepath = os.path.join(imagenet_dir, f"{index}.png")
                if not os.path.isfile(imagepath):
                    image = np.array(f['val/images'][index])
                    Image.fromarray(image).save(imagepath)
                filepaths.append(imagepath)

        return filepaths

    def _get_batch_activations(self, inputs, layer_names, batch_size):
        inputs, num_padding = self._pad(inputs, batch_size)
        preprocessed_inputs = self.preprocess(inputs)
        activations = self.get_activations(preprocessed_inputs, layer_names)
        assert isinstance(activations, OrderedDict)
        activations = self._unpad(activations, num_padding)
        return activations

    def _package(self, layer_activations, stimuli_paths):
        activations = list(layer_activations.values())
        shapes = [a.shape for a in activations]
        self._logger.debug('Activations shapes: {}'.format(shapes))
        # layer x images x activations --> images x (layer x activations)
        activations = np.concatenate(activations, axis=-1)
        assert activations.shape[0] == len(stimuli_paths)
        assert activations.shape[1] == np.sum([np.prod(shape[1:]) for shape in shapes])
        layers = []
        for layer, shape in zip(layer_activations.keys(), shapes):
            repeated_layer = [layer] * np.prod(shape[1:])
            layers += repeated_layer
        model_assembly = NeuroidAssembly(
            activations,
            coords={'stimulus_path': stimuli_paths,
                    'neuroid_id': ('neuroid', list(range(activations.shape[1]))),
                    'layer': ('neuroid', layers)},
            dims=['stimulus_path', 'neuroid']
        )
        return model_assembly

    def _pad(self, batch_images, batch_size):
        num_images = len(batch_images)
        # try:  # `len` for numpy arrays and lists (of filepaths)
        #     num_images = len(batch_images)
        # except TypeError:  # `.shape` for tensors without `__len__` (e.g. in TensorFlow)
        #     num_images = batch_images.shape[0]
        if num_images % batch_size == 0:
            return batch_images, 0
        num_padding = batch_size - (num_images % batch_size)
        padding = np.repeat(batch_images[-1:], repeats=num_padding, axis=0)
        return np.concatenate((batch_images, padding)), num_padding

    def _unpad(self, layer_activations, num_padding):
        return self._change_layer_activations(layer_activations, lambda values: values[:-num_padding or None])

    def _change_layer_activations(self, layer_activations, change_function, keep_name=False, multithread=False):
        if not multithread:
            map_fnc = map
        else:
            pool = ThreadPool()
            map_fnc = pool.map

        def apply_change(layer_values):
            layer, values = layer_values
            values = change_function(values) if not keep_name else change_function(layer, values)
            return layer, values

        results = map_fnc(apply_change, layer_activations.items())
        results = OrderedDict(results)
        if multithread:
            pool.close()
        return results


def flatten(layer_name, layer_output):
    return layer_output.reshape(layer_output.shape[0], -1)
