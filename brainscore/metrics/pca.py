import logging
import os

import h5py
import numpy as np
from PIL import Image
from result_caching import store
from sklearn.decomposition import PCA as PCAImpl
from tqdm import tqdm

from brainio_base.assemblies import walk_coords, merge_data_arrays
from brainscore.utils import fullname, s3


# TODO: every layer is computed separately right now. but can't run them together in brain_transformation
#  since that would store non-PCA'd activations


class PCA:
    DEFAULT_BATCH_SIZE = 64

    def __init__(self, n_components):
        self._logger = logging.getLogger(fullname(self))
        self._n_components = n_components

    def __call__(self, model_identifier, model, stimuli, batch_size=DEFAULT_BATCH_SIZE):
        return self._call(model_identifier=model_identifier, stimuli_identifier=stimuli.name,
                          model=model, stimuli=stimuli, batch_size=batch_size)

    @store(identifier_ignore=['model', 'stimuli', 'batch_size'])
    def _call(self, model_identifier, stimuli_identifier, model, stimuli, batch_size=DEFAULT_BATCH_SIZE):
        pca = self._initialize_pca(model_identifier, model, n_components=self._n_components)
        if not pca:
            return model(stimuli)
        pca_assembly = []
        for batch_start in tqdm(range(0, len(stimuli), batch_size), unit_scale=batch_size, desc="pca batch"):
            batch_end = min(batch_start + batch_size, len(stimuli))
            batch_stimuli = stimuli[batch_start:batch_end]
            batch_assembly = model(batch_stimuli)
            batch_assembly = batch_assembly.transpose('presentation', 'neuroid')
            pca_values = pca.transform(batch_assembly)
            pca_batch_assembly = self.package(pca_values, batch_assembly)
            pca_assembly.append(pca_batch_assembly)
        pca_assembly = merge_data_arrays(pca_assembly)
        assert set(pca_assembly['image_id'].values) == set(stimuli['image_id'].values)
        return pca_assembly

    @store(identifier_ignore=['model'])
    def _initialize_pca(self, model_identifier, model, n_components):
        self._logger.debug('Retrieving ImageNet activations')
        imagenet_paths = _get_imagenet_val(num_images=n_components)
        imagenet_assembly = model(imagenet_paths)

        self._logger.debug('Computing ImageNet principal components')
        if len(imagenet_assembly['neuroid']) <= n_components:
            self._logger.debug(f"Not computing principal components for {model_identifier} activations "
                               f"as shape {imagenet_assembly.shape} is small enough already")
            pca = None
        else:
            pca = PCAImpl(n_components=n_components, random_state=0)
            pca.fit(imagenet_assembly.transpose('stimulus_path', 'neuroid'))
        return pca

    def package(self, pca_values, source_assembly):
        coords = {coord: (dims, values) for coord, dims, values in walk_coords(source_assembly['presentation'])}
        coords['neuroid_id'] = 'neuroid', list(range(pca_values.shape[1]))
        # attach single-value neuroid meta
        for coord, dims, values in walk_coords(source_assembly['neuroid']):
            unique_value = np.unique(values)
            if len(unique_value) == 1:
                coords[coord] = dims, np.repeat(unique_value, pca_values.shape[1])
        return type(source_assembly)(pca_values, coords=coords, dims=source_assembly.dims)


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

    framework_home = os.path.expanduser(os.getenv('BSC_HOME', '~/.brain-score'))
    imagenet_filepath = os.getenv('BSC_IMAGENET_PATH', os.path.join(framework_home, 'imagenet2012.hdf5'))
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
