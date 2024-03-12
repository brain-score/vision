import os

import functools
import logging
from collections import OrderedDict
from multiprocessing.pool import ThreadPool

import numpy as np

from brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.utils import fullname
from result_caching import store_xarray
from .inferencer import Inferencer


class ActivationsExtractor:
    """A wrapper for the inferencer to provide additional functionalities.
    
        Specifically, it converts the stimulus_set to a list of paths and then calls the inferencer to get the activations.
        Then, it stores the activations in a NeuroidAssembly on the and returns it.
    """
    def __init__(
            self, 
            inferencer : Inferencer,
            identifier=False,
        ):
        self._logger = logging.getLogger(fullname(self))

        self._identifier = identifier
        self.inferencer = inferencer
        self._stimulus_set_hooks = {}
        self._batch_activations_hooks = {}

    @property
    def identifier(self):
        if self._identifier:
            return f"{self._identifier}@{self.inferencer.identifier}"
        else:
            return False

    def insert_attrs(self, wrapper):
        wrapper.from_stimulus_set = self.from_stimulus_set
        wrapper.from_paths = self.from_paths
        wrapper.register_batch_activations_hook = self.register_batch_activations_hook
        wrapper.register_stimulus_set_hook = self.register_stimulus_set_hook

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
        if stimuli_identifier is None and hasattr(stimulus_set, 'identifier'):
            stimuli_identifier = stimulus_set.identifier
        for hook in self._stimulus_set_hooks.copy().values():  # copy to avoid stale handles
            stimulus_set = hook(stimulus_set)
        stimuli_paths = [str(stimulus_set.get_stimulus(stimulus_id)) for stimulus_id in stimulus_set['stimulus_id']]
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
        if len(layers) == 0:
            raise ValueError("No layers passed to retrieve activations from")
        return self.inferencer(stimuli_paths, layers)

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
    stimulus_paths = [str(stimulus_set.get_stimulus(stimulus_id)) for stimulus_id in stimulus_set['stimulus_id']]
    stimulus_paths = [lstrip_local(path) for path in stimulus_paths]
    assembly_paths = [lstrip_local(path) for path in assembly['stimulus_path'].values]
    assert (np.array(assembly_paths) == np.array(stimulus_paths)).all()
    assembly['stimulus_path'] = stimulus_set['stimulus_id'].values
    assembly = assembly.rename({'stimulus_path': 'stimulus_id'})
    for column in stimulus_set.columns:
        assembly[column] = 'stimulus_id', stimulus_set[column].values
    assembly = assembly.stack(presentation=('stimulus_id',))
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


def flatten(layer_output, from_index=1, return_index=False):
    flattened = layer_output.reshape(*layer_output.shape[:from_index], -1)
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

    index = cartesian_product_broadcasted(*[np.arange(s, dtype='int') for s in layer_output.shape[from_index:]])
    return flattened, index
