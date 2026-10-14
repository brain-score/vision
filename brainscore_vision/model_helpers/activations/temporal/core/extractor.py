import os
import tempfile

import functools
import logging
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
from typing import List, Any, Callable

import numpy as np
from PIL import Image as PILImage

from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.assemblies import DataAssembly
from brainscore_vision.model_helpers.activations.cache_key import (
    model_cache_identifier, stimulus_set_cache_identifier)
from brainscore_vision.model_helpers.activations.core import NO_RESOLUTION, RESOLUTION_FORMAT
from brainscore_vision.model_helpers.utils import fullname
from result_caching import store_xarray
from .inferencer import Inferencer
from ..inputs import Stimulus, Video


# Preprocessing resolution, for the stored-activations cache key
# ---------------------------------------------------------------
# The same gap the image path closed (see `activations/core.py`): the stored
# key names the model and the stimulus set, and neither says what the model is
# actually fed. Two commitments that share an identifier but preprocess to
# different resolutions are served each other's activations, no preprocessing
# runs, and the benchmark returns a plausible score for a resolution the model
# never saw.
#
# The image path's probe cannot be reused as it stands: a temporal
# preprocessing is handed a single `Stimulus`, not a list of paths -- see
# `core/executor.py`, where the executor maps it over stimuli one at a time.
# The sentinel and the output format are imported from there rather than
# restated, so the two halves of the key cannot drift apart.

# Deliberately neither square nor a common model input size, for the same
# reason as the image path's probe image: a preprocessing that hands its input
# back unchanged must not be mistakable for one that resizes.
_PROBE_FRAME_SHAPE = (137, 211, 3)  # height, width, RGB

# Both of these reach the key: the frame count is part of the fingerprint
# (`3x10x112x112` carries T=10 from here, not from the model), so changing
# either invalidates every stored temporal cache entry.
_PROBE_FPS = 10.
# A full second, not the handful of frames the spatial part of the fingerprint
# would need. `Video.set_fps` asserts that one frame period fits inside the
# clip (`1000/fps <= duration`), so a short probe silently fails for every
# preprocessing that resamples to a low frame rate -- at 300ms, anything below
# 3.33 fps. One second admits any fps >= 1, which covers the realistic range.
_PROBE_DURATION = 1000.  # ms; with _PROBE_FPS, ten frames

_logger = logging.getLogger(__name__)
_probe_frame_path = None


def _probe_frame():
    """Path to the synthetic probe frame, written once per process."""
    global _probe_frame_path
    if _probe_frame_path is None or not os.path.isfile(_probe_frame_path):
        pixels = np.random.RandomState(0).randint(0, 256, _PROBE_FRAME_SHAPE, dtype=np.uint8)
        directory = tempfile.mkdtemp(prefix='brainscore-temporal-resolution-probe-')
        path = os.path.join(directory, 'probe.png')
        PILImage.fromarray(pixels).save(path)
        _probe_frame_path = path
    return _probe_frame_path


def _probe_stimulus(stimulus_type):
    """A synthetic stimulus of `stimulus_type`, built without decoding a video.

    A still frame is enough: `Video.from_img_path` returns a `VideoFromImage`,
    which repeats that frame and so never reaches `decord`. Writing a real clip
    would be the obvious alternative and was measured first -- but every codec
    available here rounds the frame size to even numbers (137x211 is read back
    as 136x210), and the exact spatial size is the one thing this probe exists
    to report.
    """
    path = _probe_frame()
    if isinstance(stimulus_type, type) and issubclass(stimulus_type, Video):
        video = stimulus_type.from_img_path(path, duration=_PROBE_DURATION, fps=_PROBE_FPS)
        # `Video.__getattribute__` resolves the `_original_*` stats lazily
        # through `cv2.VideoCapture`, which cannot open a still image and logs
        # two ffmpeg errors before falling back to it. Filling them in here
        # keeps cv2 out of the probe entirely; `Video.copy` populates the same
        # three attributes the same way.
        video._original_fps = _PROBE_FPS
        video._original_duration = _PROBE_DURATION
        video._original_size = video.frame_size
        return video
    return stimulus_type.from_path(path)


def probe_preprocessing_resolution(preprocess, stimulus_type, probe_stimuli=None):
    """Measure what `preprocess` feeds the model, as a cache-key component.

    Calls `preprocess` the way `BatchExecutor.execute` does -- with one
    stimulus -- and reports the shape and dtype of what comes back, e.g.
    ``'3x10x112x112-float32'`` for a (C, T, H, W) tensor. There is no batch
    dimension to drop: unlike the image path, the temporal preprocessing is
    called per stimulus.

    :param preprocess: the inferencer's preprocessing callable.
    :param stimulus_type: the inferencer's stimulus type, which decides what
        kind of synthetic stimulus the probe is built from.
    :param probe_stimuli: stimuli to probe with, for tests that need to pin the
        input; defaults to a single synthetic stimulus.
    :return: the key component, or `NO_RESOLUTION` if nothing could be measured.

    Never raises. This runs while an extractor is being constructed, for every
    temporal model in the registry; a preprocessing that cannot be probed must
    cost a coarser cache key, never a model that cannot be instantiated.
    """
    try:
        stimuli = list(probe_stimuli) if probe_stimuli is not None else [_probe_stimulus(stimulus_type)]
        if len(stimuli) == 0:
            return NO_RESOLUTION
        preprocessed = preprocess(stimuli[0])
        # A preprocessing that returns a container of per-item tensors carries
        # the shape on its elements, as on the image path.
        if isinstance(preprocessed, (list, tuple)):
            if len(preprocessed) == 0:
                return NO_RESOLUTION
            preprocessed = preprocessed[0]
        shape = getattr(preprocessed, 'shape', None)
        dtype = getattr(preprocessed, 'dtype', None)
        # An identity preprocessing hands the `Stimulus` straight back, which
        # has neither, and there is no resolution to speak of.
        if shape is None or dtype is None:
            return NO_RESOLUTION
        # `torch.float32` and numpy's `float32` must not key differently.
        component = 'x'.join(str(int(d)) for d in tuple(shape)) + '-' + str(dtype).replace('torch.', '')
        return component if RESOLUTION_FORMAT.match(component) else NO_RESOLUTION
    except Exception as e:  # noqa: BLE001 - a failed probe must not fail a run
        _logger.debug(f"Could not probe the preprocessing resolution ({type(e).__name__}: {e}), "
                      f"caching under `{NO_RESOLUTION}`")
        return NO_RESOLUTION


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
        # Probed once here rather than per request: it costs milliseconds and
        # cannot change for the life of the extractor. After
        # `set_visual_degrees`, so the probe sees the configuration the
        # stimuli will actually be preprocessed under.
        self._resolution = probe_preprocessing_resolution(
            self.inferencer.preprocessing, self.inferencer.stimulus_type)

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
        # Identifiers alone do not distinguish plugin revisions, so a cache
        # keyed on them would serve activations from code that no longer
        # exists. Append a content revision to each. These values are used
        # *only* to build the cache key -- _from_paths_stored ignores them
        # otherwise -- so the returned assembly is unaffected. Either is None
        # when a revision was required but unresolvable, which means this
        # request must not be cached at all.
        #
        # The revision is looked up under `self._identifier` rather than
        # `self.identifier`: the latter carries the inferencer configuration
        # after an `@` and is registered nowhere, so resolving it finds no
        # plugin and would refuse to cache every temporal model. The full
        # identifier still goes into the key -- the inferencer configuration
        # changes the activations.
        cache_identifier = (model_cache_identifier(self.identifier,
                                                   lookup_identifier=self._identifier)
                            if self.identifier else None)
        cache_stimuli_identifier = (stimulus_set_cache_identifier(stimuli_identifier)
                                    if stimuli_identifier else None)
        if cache_identifier and cache_stimuli_identifier:
            fnc = functools.partial(self._from_paths_stored,
                                    identifier=cache_identifier,
                                    stimuli_identifier=cache_stimuli_identifier,
                                    # getattr: extractors built with `__new__`
                                    # never ran `__init__` and have no probe.
                                    resolution=getattr(self, '_resolution', None) or NO_RESOLUTION)
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
    def _from_paths_stored(self, identifier, layers, stimuli_identifier, resolution, stimuli_paths):
        """`resolution` is used *only* to build the cache key -- like `identifier`
        and `stimuli_identifier`, it is not forwarded into `_from_paths`. It sits
        behind `stimuli_identifier` for the same reason as on the image path:
        `brainscore_vision.score_model` prints a cache-cleanup hint globbing
        `identifier=...,stimuli_identifier=*.pkl`, which a component inserted
        ahead of `stimuli_identifier` would silently stop matching.
        """
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
