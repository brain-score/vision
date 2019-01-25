import logging
import os

import h5py
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

from model_tools.utils import fullname, s3
from result_caching import store_dict


class LayerPCA:
    def __init__(self, activations_extractor, n_components):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._n_components = n_components
        self._layer_pcas = {}

    def __call__(self, batch_activations):
        self._ensure_initialized(batch_activations.keys())

        def apply_pca(layer, activations):
            pca = self._layer_pcas[layer]
            activations = flatten(activations)
            if pca is None:
                return activations
            return pca.transform(activations)

        from model_tools.activations.core import change_dict
        return change_dict(batch_activations, apply_pca, keep_name=True, multithread=False)

    def _ensure_initialized(self, layers):
        missing_layers = [layer for layer in layers if layer not in self._layer_pcas]
        if len(missing_layers) == 0:
            return
        layer_pcas = self._pcas(identifier=self._extractor.identifier, layers=missing_layers,
                                n_components=self._n_components)
        self._layer_pcas = {**self._layer_pcas, **layer_pcas}

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _pcas(self, identifier, layers, n_components):
        self._logger.debug('Retrieving ImageNet activations')
        imagenet_paths = _get_imagenet_val(num_images=n_components)
        self.handle.disable()
        imagenet_activations = self._extractor(imagenet_paths, layers=layers)
        imagenet_activations = {layer: imagenet_activations.sel(layer=layer).values
                                for layer in np.unique(imagenet_activations['layer'])}
        assert len(set(activations.shape[0] for activations in imagenet_activations.values())) == 1, "stimuli differ"
        self.handle.enable()

        self._logger.debug('Computing ImageNet principal components')
        progress = tqdm(total=len(imagenet_activations), desc="layer principal components")

        def init_and_progress(layer, activations):
            activations = flatten(activations)
            if activations.shape[1] <= n_components:
                self._logger.debug(f"Not computing principal components for {layer} "
                                   f"activations {activations.shape} as shape is small enough already")
                pca = None
            else:
                pca = PCA(n_components=n_components)
                pca.fit(activations)
            progress.update(1)
            return pca

        from model_tools.activations.core import change_dict
        layer_pcas = change_dict(imagenet_activations, init_and_progress, keep_name=True, multithread=True)
        progress.close()
        return layer_pcas


def _get_imagenet_val(num_images):
    _logger = logging.getLogger(fullname(_get_imagenet_val))
    num_classes = 1000
    num_images_per_class = (num_images - 1) // num_classes
    base_indices = np.arange(num_images_per_class).astype(int)
    indices = []
    for i in range(num_classes):
        indices.extend(50 * i + base_indices)
    for i in range((num_images - 1) % num_classes + 1):
        indices.extend(50 * i + np.array([num_images_per_class]).astype(int))

    framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.model-tools'))
    imagenet_filepath = os.getenv('CM_IMAGENET_PATH', os.path.join(framework_home, 'imagenet2012.hdf5'))
    imagenet_dir = f"{imagenet_filepath}-files"
    os.makedirs(imagenet_dir, exist_ok=True)

    if not os.path.isfile(imagenet_filepath):
        os.makedirs(os.path.dirname(imagenet_filepath), exist_ok=True)
        _logger.debug(f"Downloading ImageNet validation to {imagenet_filepath}")
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


def flatten(layer_output):
    return layer_output.reshape(layer_output.shape[0], -1)
