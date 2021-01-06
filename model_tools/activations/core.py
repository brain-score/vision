import copy
import os

import functools
import logging
from collections import OrderedDict
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm import tqdm

from brainio_base.assemblies import NeuroidAssembly, walk_coords
from brainio_base.stimuli import StimulusSet
from model_tools.utils import fullname
from result_caching import store_xarray


class Defaults:
    batch_size = 64


class ActivationsExtractorHelper:
    def __init__(self, get_activations, preprocessing, identifier=False, batch_size=Defaults.batch_size):
        """
        :param identifier: an activations identifier for the stored results file. False to disable saving.
        """
        self._logger = logging.getLogger(fullname(self))

        self._batch_size = batch_size
        self.identifier = identifier
        self.get_activations = get_activations
        self.preprocess = preprocessing or (lambda x: x)
        self._stimulus_set_hooks = {}
        self._batch_activations_hooks = {}

    def __call__(self, stimuli, layers, stimuli_identifier=None):
        """
        :param stimuli_identifier: a stimuli identifier for the stored results file. False to disable saving.
        """
        if isinstance(stimuli, StimulusSet):
            return self.from_stimulus_set(stimulus_set=stimuli, layers=layers, stimuli_identifier=stimuli_identifier)
        else:
            return self.from_paths(stimuli_paths=stimuli, layers=layers, stimuli_identifier=stimuli_identifier)

    def from_stimulus_set(self, stimulus_set, layers, stimuli_identifier=None):
        """
        :param stimuli_identifier: a stimuli identifier for the stored results file.
            False to disable saving. None to use `stimulus_set.identifier`
        """
        if stimuli_identifier is None:
            stimuli_identifier = stimulus_set.identifier
        for hook in self._stimulus_set_hooks.copy().values():  # copy to avoid stale handles
            stimulus_set = hook(stimulus_set)
        stimuli_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
        activations = self.from_paths(stimuli_paths=stimuli_paths, layers=layers, stimuli_identifier=stimuli_identifier)
        activations = attach_stimulus_set_meta(activations, stimulus_set)
        return activations

    def from_paths(self, stimuli_paths, layers, stimuli_identifier=None):
        if layers is None:
            layers = ['logits']
        if self.identifier and stimuli_identifier:
            fnc = functools.partial(self._from_paths_stored,
                                    identifier=self.identifier, stimuli_identifier=stimuli_identifier)
        else:
            self._logger.debug(f"self.identifier `{self.identifier}` or stimuli_identifier {stimuli_identifier} "
                               f"are not set, will not store")
            fnc = self._from_paths
        # In case stimuli paths are duplicates (e.g. multiple trials), we first reduce them to only the paths that need
        # to be run individually, compute activations for those, and then expand the activations to all paths again.
        # This is done here, before storing, so that we only store the reduced activations.
        reduced_paths = self._reduce_paths(stimuli_paths)
        activations = fnc(layers=layers, stimuli_paths=reduced_paths)
        activations = self._expand_paths(activations, original_paths=stimuli_paths)
        return activations

    @store_xarray(identifier_ignore=['stimuli_paths', 'layers'], combine_fields={'layers': 'layer'})
    def _from_paths_stored(self, identifier, layers, stimuli_identifier, stimuli_paths):
        return self._from_paths(layers=layers, stimuli_paths=stimuli_paths)

    def _from_paths(self, layers, stimuli_paths):
        self._logger.info('Running stimuli')
        layer_activations = self._get_activations_batched(stimuli_paths, layers=layers, batch_size=self._batch_size)
        self._logger.info('Packaging into assembly')
        return self._package(layer_activations, stimuli_paths)

    def _reduce_paths(self, stimuli_paths):
        return list(set(stimuli_paths))

    def _expand_paths(self, activations, original_paths):
        activations_paths = activations['stimulus_path'].values
        argsort_indices = np.argsort(activations_paths)
        sorted_x = activations_paths[argsort_indices]
        sorted_index = np.searchsorted(sorted_x, original_paths)
        index = [argsort_indices[i] for i in sorted_index]
        return activations[{'stimulus_path': index}]


    def register_batch_activations_hook(self, hook):
        r"""
        The hook will be called every time a batch of activations is retrieved.
        The hook should have the following signature::

            hook(batch_activations) -> batch_activations

        The hook should return new batch_activations which will be used in place of the previous ones.
        """

        handle = HookHandle(self._batch_activations_hooks)
        self._batch_activations_hooks[handle.id] = hook
        return handle

    def register_stimulus_set_hook(self, hook):
        r"""
        The hook will be called every time before a stimulus set is processed.
        The hook should have the following signature::

            hook(stimulus_set) -> stimulus_set

        The hook should return a new stimulus_set which will be used in place of the previous one.
        """

        handle = HookHandle(self._stimulus_set_hooks)
        self._stimulus_set_hooks[handle.id] = hook
        return handle

    def _get_activations_batched(self, paths, layers, batch_size):
        layer_activations = None
        for batch_start in tqdm(range(0, len(paths), batch_size), unit_scale=batch_size, desc="activations"):
            batch_end = min(batch_start + batch_size, len(paths))
            batch_inputs = paths[batch_start:batch_end]
            batch_activations = self._get_batch_activations(batch_inputs, layer_names=layers, batch_size=batch_size)
            for hook in self._batch_activations_hooks.copy().values():  # copy to avoid handle re-enabling messing with the loop
                batch_activations = hook(batch_activations)

            if layer_activations is None:
                layer_activations = copy.copy(batch_activations)
            else:
                for layer_name, layer_output in batch_activations.items():
                    layer_activations[layer_name] = np.concatenate((layer_activations[layer_name], layer_output))

        return layer_activations

    def _get_batch_activations(self, inputs, layer_names, batch_size):
        inputs, num_padding = self._pad(inputs, batch_size)
        preprocessed_inputs = self.preprocess(inputs)
        activations = self.get_activations(preprocessed_inputs, layer_names)
        assert isinstance(activations, OrderedDict)
        activations = self._unpad(activations, num_padding)
        return activations

    def _pad(self, batch_images, batch_size):
        num_images = len(batch_images)
        if num_images % batch_size == 0:
            return batch_images, 0
        num_padding = batch_size - (num_images % batch_size)
        padding = np.repeat(batch_images[-1:], repeats=num_padding, axis=0)
        return np.concatenate((batch_images, padding)), num_padding

    def _unpad(self, layer_activations, num_padding):
        return change_dict(layer_activations, lambda values: values[:-num_padding or None])

    def _package(self, layer_activations, stimuli_paths):
        shapes = [a.shape for a in layer_activations.values()]
        self._logger.debug('Activations shapes: {}'.format(shapes))
        self._logger.debug("Packaging individual layers")
        layer_assemblies = [self._package_layer(single_layer_activations, layer=layer, stimuli_paths=stimuli_paths) for
                            layer, single_layer_activations in tqdm(layer_activations.items(), desc='layer packaging')]
        # merge manually instead of using merge_data_arrays since `xarray.merge` is very slow with these large arrays
        # complication: (non)neuroid_coords are taken from the structure of layer_assemblies[0] i.e. the 1st assembly;
        # using these names/keys for all assemblies results in KeyError if the first layer contains flatten_coord_names
        # (see _package_layer) not present in later layers, e.g. first layer = conv, later layer = transformer layer
        self._logger.debug("Merging layer assemblies")
        model_assembly = np.concatenate([a.values for a in layer_assemblies],
                                        axis=layer_assemblies[0].dims.index('neuroid'))
        nonneuroid_coords = {coord: (dims, values) for coord, dims, values in walk_coords(layer_assemblies[0])
                             if set(dims) != {'neuroid'}}
        neuroid_coords = {coord: [dims, values] for coord, dims, values in walk_coords(layer_assemblies[0])
                          if set(dims) == {'neuroid'}}
        for layer_assembly in layer_assemblies[1:]:
            for coord in neuroid_coords:
                neuroid_coords[coord][1] = np.concatenate((neuroid_coords[coord][1], layer_assembly[coord].values))
            assert layer_assemblies[0].dims == layer_assembly.dims
            for dim in set(layer_assembly.dims) - {'neuroid'}:
                for coord in layer_assembly[dim].coords:
                    assert (layer_assembly[coord].values == nonneuroid_coords[coord][1]).all()
        neuroid_coords = {coord: (dims_values[0], dims_values[1])  # re-package as tuple instead of list for xarray
                          for coord, dims_values in neuroid_coords.items()}
        model_assembly = type(layer_assemblies[0])(model_assembly, coords={**nonneuroid_coords, **neuroid_coords},
                                                   dims=layer_assemblies[0].dims)
        return model_assembly

    def _package_layer(self, layer_activations, layer, stimuli_paths):
        assert layer_activations.shape[0] == len(stimuli_paths)
        activations, flatten_indices = flatten(layer_activations, return_index=True)  # collapse for single neuroid dim
        assert flatten_indices.shape[1] in [1, 2, 3]
        # see comment in _package for an explanation why we cannot simply have 'channel' for the FC layer
        if flatten_indices.shape[1] == 1:    # FC
            flatten_coord_names = ['channel', 'channel_x', 'channel_y']
        elif flatten_indices.shape[1] == 2:  # Transformer
            flatten_coord_names = ['channel', 'embedding']
        elif flatten_indices.shape[1] == 3:  # 2DConv
            flatten_coord_names = ['channel', 'channel_x', 'channel_y']
        flatten_coords = {flatten_coord_names[i]: [sample_index[i] if i < flatten_indices.shape[1] else np.nan for sample_index in flatten_indices]
                          for i in range(len(flatten_coord_names))}
        layer_assembly = NeuroidAssembly(
            activations,
            coords={**{'stimulus_path': stimuli_paths,
                       'neuroid_num': ('neuroid', list(range(activations.shape[1]))),
                       'model': ('neuroid', [self.identifier] * activations.shape[1]),
                       'layer': ('neuroid', [layer] * activations.shape[1]),
                       },
                    **{coord: ('neuroid', values) for coord, values in flatten_coords.items()}},
            dims=['stimulus_path', 'neuroid']
        )
        neuroid_id = [".".join([f"{value}" for value in values]) for values in zip(*[
            layer_assembly[coord].values for coord in ['model', 'layer', 'neuroid_num']])]
        layer_assembly['neuroid_id'] = 'neuroid', neuroid_id
        return layer_assembly

    def insert_attrs(self, wrapper):
        wrapper.from_stimulus_set = self.from_stimulus_set
        wrapper.from_paths = self.from_paths
        wrapper.register_batch_activations_hook = self.register_batch_activations_hook
        wrapper.register_stimulus_set_hook = self.register_stimulus_set_hook


def change_dict(d, change_function, keep_name=False, multithread=False):
    if not multithread:
        map_fnc = map
    else:
        pool = ThreadPool()
        map_fnc = pool.map

    def apply_change(layer_values):
        layer, values = layer_values
        values = change_function(values) if not keep_name else change_function(layer, values)
        return layer, values

    results = map_fnc(apply_change, d.items())
    results = OrderedDict(results)
    if multithread:
        pool.close()
    return results


def lstrip_local(path):
    parts = path.split(os.sep)
    try:
        start_index = parts.index('.brainio')
    except ValueError:  # not in list -- perhaps custom directory
        return path
    path = os.sep.join(parts[start_index:])
    return path


def attach_stimulus_set_meta(assembly, stimulus_set):
    stimulus_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
    stimulus_paths = [lstrip_local(path) for path in stimulus_paths]
    assembly_paths = [lstrip_local(path) for path in assembly['stimulus_path'].values]
    assert (np.array(assembly_paths) == np.array(stimulus_paths)).all()
    assembly['stimulus_path'] = stimulus_set['image_id'].values
    assembly = assembly.rename({'stimulus_path': 'image_id'})
    for column in stimulus_set.columns:
        assembly[column] = 'image_id', stimulus_set[column].values
    assembly = assembly.stack(presentation=('image_id',))
    return assembly


class HookHandle:
    next_id = 0

    def __init__(self, hook_dict):
        self.hook_dict = hook_dict
        self.id = HookHandle.next_id
        HookHandle.next_id += 1
        self._saved_hook = None

    def remove(self):
        hook = self.hook_dict[self.id]
        del self.hook_dict[self.id]
        return hook

    def disable(self):
        self._saved_hook = self.remove()

    def enable(self):
        self.hook_dict[self.id] = self._saved_hook
        self._saved_hook = None


def flatten(layer_output, return_index=False):
    flattened = layer_output.reshape(layer_output.shape[0], -1)
    if not return_index:
        return flattened

    def cartesian_product_broadcasted(*arrays):
        """
        http://stackoverflow.com/a/11146645/190597
        """
        broadcastable = np.ix_(*arrays)
        broadcasted = np.broadcast_arrays(*broadcastable)
        dtype = np.result_type(*arrays)
        rows, cols = functools.reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
        out = np.empty(rows * cols, dtype=dtype)
        start, end = 0, rows
        for a in broadcasted:
            out[start:end] = a.reshape(-1)
            start, end = end, end + rows
        return out.reshape(cols, rows).T

    index = cartesian_product_broadcasted(*[np.arange(s, dtype='int') for s in layer_output.shape[1:]])
    return flattened, index
