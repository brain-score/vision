import copy
import functools
import logging
from collections import OrderedDict
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm import tqdm

from brainio_base.assemblies import NeuroidAssembly
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
        self._batch_hooks = {}

    def __call__(self, stimuli, layers, stimuli_identifier=False):
        """
        :param stimuli_identifier: a stimuli identifier for the stored results file. False to disable saving.
        """
        if isinstance(stimuli, StimulusSet):
            return self.from_stimulus_set(stimulus_set=stimuli, layers=layers)
        else:
            return self.from_paths(stimuli_paths=stimuli, layers=layers, stimuli_identifier=stimuli_identifier)

    def from_stimulus_set(self, stimulus_set, layers, stimuli_identifier=False):
        """
        :param stimuli_identifier: a stimuli identifier for the stored results file.
            False to disable saving. None to use `stimulus_set.name`
        """
        if stimuli_identifier is None:
            stimuli_identifier = stimulus_set.name
        stimuli_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
        activations = self.from_paths(stimuli_paths=stimuli_paths, layers=layers, stimuli_identifier=stimuli_identifier)
        activations = attach_stimulus_set_meta(activations, stimulus_set)
        return activations

    def from_paths(self, stimuli_paths, layers, stimuli_identifier=False):
        if self.identifier and stimuli_identifier:
            fnc = functools.partial(self._from_paths_stored,
                                    identifier=self.identifier, stimuli_identifier=stimuli_identifier)
        else:
            self._logger.debug(f"self.identifier `{self.identifier}` or stimuli_identifier {stimuli_identifier} "
                               f"are not set, will not store")
            fnc = self._from_paths
        return fnc(layers=layers, stimuli_paths=stimuli_paths)

    @store_xarray(identifier_ignore=['stimuli_paths', 'layers'], combine_fields={'layers': 'layer'})
    def _from_paths_stored(self, identifier, layers, stimuli_identifier, stimuli_paths):
        return self._from_paths(layers=layers, stimuli_paths=stimuli_paths)

    def _from_paths(self, layers, stimuli_paths):
        self._logger.info('Running stimuli')
        layer_activations = self._get_activations_batched(stimuli_paths, layers=layers, batch_size=self._batch_size)
        self._logger.info('Packaging into assembly')
        return self._package(layer_activations, stimuli_paths)

    def register_batch_hook(self, hook):
        r"""
        The hook will be called every time a batch of activations is retrieved.
        The hook should have the following signature::

            hook(batch_activations) -> batch_activations

        The hook should return new batch_activations which will be used in place of the previous ones.
        """

        handle = HookHandle(self._batch_hooks)
        self._batch_hooks[handle.id] = hook
        return handle

    def _get_activations_batched(self, inputs, layers, batch_size):
        layer_activations = None
        for batch_start in tqdm(range(0, len(inputs), batch_size), unit_scale=batch_size, desc="activations"):
            batch_end = min(batch_start + batch_size, len(inputs))
            self._logger.debug('Batch %d->%d/%d', batch_start, batch_end, len(inputs))
            batch_inputs = inputs[batch_start:batch_end]
            batch_activations = self._get_batch_activations(batch_inputs, layer_names=layers, batch_size=batch_size)
            for hook in self._batch_hooks.copy().values():  # copy to avoid handle re-enabling messing with the loop
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
        activations = list(layer_activations.values())
        shapes = [a.shape for a in activations]
        self._logger.debug('Activations shapes: {}'.format(shapes))
        activations = [flatten(single_layer_activations) for single_layer_activations in activations]  # collapse
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

    def insert_attrs(self, wrapper):
        wrapper.identifier = self.identifier
        wrapper.from_stimulus_set = self.from_stimulus_set
        wrapper.from_paths = self.from_paths
        wrapper.register_batch_hook = self.register_batch_hook


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


def attach_stimulus_set_meta(assembly, stimulus_set):
    stimulus_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
    assert all(assembly['stimulus_path'] == stimulus_paths)
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


def flatten(layer_output):
    return layer_output.reshape(layer_output.shape[0], -1)
