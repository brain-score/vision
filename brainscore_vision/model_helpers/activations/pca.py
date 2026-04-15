import h5py
import logging
import numpy as np
import os
from PIL import Image
from result_caching import store_dict
from sklearn.decomposition import PCA
from tqdm import tqdm

from brainscore_vision.model_helpers.activations.core import flatten, change_dict
from brainscore_vision.model_helpers.utils import fullname, s3


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

        return change_dict(batch_activations, apply_pca, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

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

        self._logger.debug('Computing ImageNet principal components')
        layer_pcas = {}
        chunks = self._chunk_layers(layers, imagenet_paths, n_components)
        progress = tqdm(total=len(layers), desc="layer principal components")
        for chunk in chunks:
            activations = self._extractor(imagenet_paths, layers=chunk)
            for layer in chunk:
                layer_activations = activations.sel(layer=layer).values
                layer_activations = flatten(layer_activations)
                if layer_activations.shape[1] <= n_components:
                    self._logger.debug(f"Not computing principal components for {layer} "
                                       f"activations {layer_activations.shape} "
                                       f"as shape is small enough already")
                    pca = None
                else:
                    pca = PCA(n_components=n_components, random_state=0)
                    pca.fit(layer_activations)
                layer_pcas[layer] = pca
                progress.update(1)
            del activations
        progress.close()

        self.handle.enable()
        return layer_pcas

    def _chunk_layers(self, layers, imagenet_paths, n_components):
        """Determine how many layers to extract per batch based on available memory.

        Extracts the first layer alone to measure per-layer memory cost, then
        calculates how many layers can safely fit in ~50% of remaining memory.
        Falls back to one-at-a-time if measurement fails.
        """
        if len(layers) <= 1:
            return [layers]

        try:
            import psutil
        except ImportError:
            return [[layer] for layer in layers]

        # Extract one layer to measure cost
        mem_before = psutil.Process().memory_info().rss
        probe = self._extractor(imagenet_paths, layers=[layers[0]])
        probe_activations = flatten(probe.sel(layer=layers[0]).values)
        activation_bytes = probe_activations.nbytes
        mem_after = psutil.Process().memory_info().rss
        del probe, probe_activations

        # Use RSS delta if meaningful, otherwise estimate from array size.
        # PCA workspace is roughly 3x the activation array (float64 copy + SVD).
        rss_delta = mem_after - mem_before
        per_layer_cost = max(rss_delta, activation_bytes * 3)

        available = psutil.virtual_memory().available
        # Use at most 50% of available memory, minimum 1 layer
        chunk_size = max(1, int(available * 0.5 / per_layer_cost)) if per_layer_cost > 0 else 1
        chunk_size = min(chunk_size, len(layers))
        self._logger.debug(f"PCA chunking: {per_layer_cost / 1e9:.1f} GB/layer, "
                           f"{available / 1e9:.1f} GB available, "
                           f"chunk_size={chunk_size} (of {len(layers)} layers)")

        # First layer was already extracted as probe — include it in first chunk
        # but it will be re-extracted (simpler than caching the probe result)
        return [layers[i:i + chunk_size] for i in range(0, len(layers), chunk_size)]

    @classmethod
    def hook(cls, activations_extractor, n_components):
        hook = LayerPCA(activations_extractor=activations_extractor, n_components=n_components)
        assert not cls.is_hooked(activations_extractor), "PCA already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())


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

    framework_home = os.path.expanduser(os.getenv('MT_HOME', '~/.model-tools'))
    imagenet_filepath = os.getenv('MT_IMAGENET_PATH', os.path.join(framework_home, 'imagenet2012.hdf5'))
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
