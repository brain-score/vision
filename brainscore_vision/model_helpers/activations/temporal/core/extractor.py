import os

import functools
import logging
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
from typing import List, Any, Callable

import numpy as np

from brainio.stimuli import StimulusSet
from brainio.assemblies import DataAssembly
from brainscore_vision.model_helpers.utils import fullname
from result_caching import store_xarray
from .inferencer import Inferencer
from ..inputs import Stimulus


# This effectively duplicates the existing non-temporal activations extractor with new functionality for temporal models,
# with minor de-coupling: now all the packaging goes to the "inferencer". 
# We hope the two can be unified again in the future or that this new wrapper will supersede the previous one
class ActivationsExtractor:
    """A wrapper for the inferencer to provide additional functionalities.
    
        Specifically, it converts the stimulus_set to a list of paths and then calls the inferencer to get the activations.
        Then, it stores the activations in a NeuroidAssembly on the and returns it.
    """
    def __init__(
            self, 
            inferencer : Inferencer,
            identifier: str = False,
            visual_degrees: float = 8,
        ):
        self._logger = logging.getLogger(fullname(self))

        self._identifier = identifier
        self.inferencer = inferencer
        self._stimulus_set_hooks = {}
        self._batch_activations_hooks = {}
        self.set_visual_degrees(visual_degrees)

    @property
    def identifier(self):
        if self._identifier:
            return f"{self._identifier}@{self.inferencer.identifier}"
        else:
            return False
        
    def set_visual_degrees(self, visual_degrees):
        self.inferencer.set_visual_degrees(visual_degrees)

    def insert_attrs(self, wrapper):
        wrapper.from_stimulus_set = self.from_stimulus_set
        wrapper.from_paths = self.from_paths
        wrapper.register_batch_activations_hook = self.register_batch_activations_hook
        wrapper.register_stimulus_set_hook = self.register_stimulus_set_hook

    def __call__(
            self, 
            stimuli : List[Stimulus], 
            layers : List[str],
            stimuli_identifier : str = None,
            number_of_trials : int = 1,
            require_variance : bool = False,
        ):
        """
        :param stimuli_identifier: a stimuli identifier for the stored results file. False to disable saving.
        """
        if number_of_trials is not None and (number_of_trials > 1 or require_variance):
            self._logger.warning("CAUTION: number_of_trials > 1 or require_variance=True is not supported yet. "
                                 "Bypassing...")
        if isinstance(stimuli, StimulusSet):
            return self.from_stimulus_set(stimulus_set=stimuli, layers=layers, stimuli_identifier=stimuli_identifier)
        else:
            return self.from_paths(stimuli_paths=stimuli, layers=layers, stimuli_identifier=stimuli_identifier)

    def from_stimulus_set(
            self, 
            stimulus_set : StimulusSet, 
            layers : List[str],
            stimuli_identifier : str = None,
        ):
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

    def from_paths(
            self, 
            stimuli_paths : List[str], 
            layers : List[str], 
            stimuli_identifier : str = None,
        ):
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
        stimuli_paths.sort()
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


def change_dict(
        d : dict, 
        change_function : Callable[[Any], Any], 
        keep_name : bool=False, 
        multithread : bool=False
    ):
    """Map a function over the values of a dictionary, recursively."""
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


def lstrip_local(path : str):
    """Strip the relative path from ".brainio". If not present, return the original path."""
    parts = path.split(os.sep)
    try:
        start_index = parts.index('.brainio')
    except ValueError:  # not in list -- perhaps custom directory
        return path
    path = os.sep.join(parts[start_index:])
    return path


def attach_stimulus_set_meta(assembly : DataAssembly, stimulus_set : Stimulus):
    """Attach all columns in the stimulus set to the assembly. 
    
    The assembly must have the "stimulus_path" coord for the "presentation" dim. 
    The stimulus set is assumed to have the same list of stimulus_path (determined by the list of stimulus_id) 
    as the assembly has. 
    """
    stimulus_paths = [str(stimulus_set.get_stimulus(stimulus_id)) for stimulus_id in stimulus_set['stimulus_id']]
    stimulus_paths = [lstrip_local(path) for path in stimulus_paths]
    assembly_paths = [lstrip_local(path) for path in assembly['stimulus_path'].values]
    assert (np.array(assembly_paths) == np.array(stimulus_paths)).all()  # check that the paths are the same
    assembly['stimulus_path'] = stimulus_set['stimulus_id'].values
    assembly = assembly.rename({'stimulus_path': 'stimulus_id'})
    for column in stimulus_set.columns:
        assembly[column] = 'stimulus_id', stimulus_set[column].values
    assembly = assembly.stack(presentation=('stimulus_id',))
    return assembly


class HookHandle:
    """A handle for enabling/disabling/removing hooks in a dictionary. 
    
    Pass a dictionary to the constructor to generate a handle in it.
    
    Example:
    >>> handle = HookHandle(hook_dict)
    >>> hook_dict[handle.id] = hook
    >>> handle.disable()
    """

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


def flatten(arr, from_index : int = 1, return_index : bool = False):
    """Flatten the array from the given index.
    
    If return_index is True, return the index of the flattened array.
    The index is a list of indices in the original array for each element.
    """
    flattened = arr.reshape(*arr.shape[:from_index], -1)
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

    index = cartesian_product_broadcasted(*[np.arange(s, dtype='int') for s in arr.shape[from_index:]])
    return flattened, index
