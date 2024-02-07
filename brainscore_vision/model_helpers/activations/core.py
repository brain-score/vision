import copy
import os
import cv2
import tempfile
from typing import List, Tuple

import functools
import logging
from collections import OrderedDict
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm.auto import tqdm
import xarray as xr

from brainio.assemblies import NeuroidAssembly, walk_coords
from brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.utils import fullname
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
        self.shifts = None  # for use with microsaccades
        self._stimulus_set_hooks = {}
        self._batch_activations_hooks = {}

    def __call__(self, stimuli, layers, stimuli_identifier=None, number_of_trials=1, require_variance=None):
        """
        :param stimuli_identifier: a stimuli identifier for the stored results file. False to disable saving.
        :param number_of_trials: An integer that determines how many repetitions of the same model performs.
        :param require_variance: A bool that asks models to output different responses to the same stimuli (i.e.,
            allows stochastic responses to identical stimuli, even in deterministic models). The current implementation
            implements this using microsaccades.
            Human microsaccade amplitude varies by who you ask, an estimate might be <0.1 deg = 360 arcsec = 6arcmin.
            The goal of microsaccades is to obtain multiple different neural activities to the same input stimulus
            from non-stochastic models. This is to improve estimates of e.g. psychophysical functions, but also other
            things. Note that microsaccades are also applied to stochastic models to make them comparable within-
            benchmark to non-stochastic models.
            In the current implementation, if `require_variance=True`, the model selects microsaccades according to
            its own microsaccade behavior (if it has implemented it), or with the base behavior of saccading in
            input pixel space with 1-pixel increments from the center of the stimulus. The base behavior thus
            maintains a fixed microsaccade distance as measured in visual angle, regardless of the model's visual angle.
            Example usage:
                require_variance = True
            More information:
            --> Rolfs 2009 "Microsaccades: Small steps on a long way" Vision Research, Volume 49, Issue 20, 15
            October 2009, Pages 2415-2441.
            --> Haddad & Steinmann 1973 "The smallest voluntary saccade: Implications for fixation" Vision
            Research Volume 13, Issue 6, June 1973, Pages 1075-1086, IN5-IN6.
            Thanks to Johannes Mehrer for initial help in implementing microsaccades.

        """
        if isinstance(stimuli, StimulusSet):
            return self.from_stimulus_set(stimulus_set=stimuli, layers=layers, stimuli_identifier=stimuli_identifier,
                                          number_of_trials=number_of_trials, require_variance=require_variance)
        else:
            return self.from_paths(stimuli_paths=stimuli,
                                   layers=layers,
                                   stimuli_identifier=stimuli_identifier,
                                   number_of_trials=number_of_trials,
                                   require_variance=require_variance)

    def from_stimulus_set(self, stimulus_set, layers, stimuli_identifier=None, number_of_trials=1,
                          require_variance=None):
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
        if require_variance:
            self.shifts = self.select_microsaccades(number_of_trials=number_of_trials)
            activations = attach_stimulus_set_meta_with_microsaccades(activations,
                                                                      stimulus_set,
                                                                      number_of_trials=number_of_trials,
                                                                      shifts=self.shifts)
        activations = attach_stimulus_set_meta(activations, stimulus_set)
        return activations

    def from_paths(self, stimuli_paths, layers, stimuli_identifier=None, number_of_trials=1, require_variance=None):
        if layers is None:
            layers = ['logits']
        if self.identifier and stimuli_identifier:
            fnc = functools.partial(self._from_paths_stored,
                                    identifier=self.identifier,
                                    stimuli_identifier=stimuli_identifier,
                                    require_variance=require_variance)
        else:
            self._logger.debug(f"self.identifier `{self.identifier}` or stimuli_identifier {stimuli_identifier} "
                               f"are not set, will not store")
            fnc = self._from_paths
        if require_variance:
            activations = fnc(layers=layers, stimuli_paths=stimuli_paths, number_of_trials=number_of_trials,
                          require_variance=require_variance)
        else:
            # In case stimuli paths are duplicates (e.g. multiple trials), we first reduce them to only the paths that need
            # to be run individually, compute activations for those, and then expand the activations to all paths again.
            # This is done here, before storing, so that we only store the reduced activations.
            reduced_paths = self._reduce_paths(stimuli_paths)
            activations = fnc(layers=layers, stimuli_paths=reduced_paths, number_of_trials=number_of_trials,
                              require_variance=require_variance)
            activations = self._expand_paths(activations, original_paths=stimuli_paths)
        return activations

    @store_xarray(identifier_ignore=['stimuli_paths', 'layers'], combine_fields={'layers': 'layer'})
    def _from_paths_stored(self, identifier, layers, stimuli_identifier,
                           stimuli_paths, number_of_trials, require_variance):
        return self._from_paths(layers=layers, stimuli_paths=stimuli_paths)

    def _from_paths(self, layers, stimuli_paths, number_of_trials=1, require_variance=None):
        if len(layers) == 0:
            raise ValueError("No layers passed to retrieve activations from")
        self._logger.info('Running stimuli')
        if require_variance:
            layer_activations = self._get_activations_batched_with_variance(stimuli_paths,
                                                                            layers=layers,
                                                                            batch_size=self._batch_size,
                                                                            number_of_trials=number_of_trials)
        else:
            layer_activations = self._get_activations_batched(stimuli_paths, layers=layers, batch_size=self._batch_size)
        self._logger.info('Packaging into assembly')
        return self._package(layer_activations, stimuli_paths, number_of_trials, require_variance)

    def _reduce_paths(self, stimuli_paths):
        return list(set(stimuli_paths))

    def _expand_paths(self, activations, original_paths):
        activations_paths = activations['stimulus_path'].values
        argsort_indices = np.argsort(activations_paths)
        sorted_x = activations_paths[argsort_indices]
        sorted_index = np.searchsorted(sorted_x, original_paths)
        index = [argsort_indices[i] for i in sorted_index]
        return activations[{'presentation': index}]

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

    def _get_activations_batched_with_variance(self, paths, layers, batch_size, number_of_trials):
        """
        This function fulfils the role of `_get_activations_batched` in the case microsaccades are needed, but
        since microsaccade activations need to be computed on an entire batch at once (to accommodate TF models),
        this function is implemented separately to avoid the mess that `_get_activations_batched` would otherwise be.

        :param number_of_trials: the number of trials that a model performs of each individual stimulus.
        """
        self.shifts = self.select_microsaccades(number_of_trials=number_of_trials)
        layer_activations = OrderedDict()
        for batch_start in tqdm(range(0, len(paths), batch_size), unit_scale=batch_size, desc="activations"):
            batch_end = min(batch_start + batch_size, len(paths))
            batch_inputs = paths[batch_start:batch_end]

            batch_activations = OrderedDict()
            # compute activations on the entire batch one shift at a time
            for shift in self.shifts:
                assert type(shift) == tuple
                temp_file_paths = self.translate_images(batch_inputs, shift)
                activations = self._get_batch_activations(temp_file_paths, layers, batch_size)
                for temp_file_path in temp_file_paths:
                    try:
                        os.remove(temp_file_path)
                    except FileNotFoundError:
                        pass
                for layer_name, layer_output in activations.items():
                    batch_activations.setdefault(layer_name, []).append(layer_output)

            # concatenate all shifts into this batch
            for layer_name, layer_outputs in batch_activations.items():
                batch_activations[layer_name] = np.concatenate(layer_outputs)

            for hook in self._batch_activations_hooks.copy().values():
                batch_activations = hook(batch_activations)

            # add this batch to layer_activations
            for layer_name, layer_output in batch_activations.items():
                layer_activations.setdefault(layer_name, []).append(layer_output)

        # fast concat all batches
        for layer_name, layer_outputs in layer_activations.items():
            layer_activations[layer_name] = np.concatenate(layer_outputs)

        return layer_activations  # this is all batches

    def _get_batch_activations(self, inputs, layer_names, batch_size):
        inputs, num_padding = self._pad(inputs, batch_size)
        preprocessed_inputs = self.preprocess(inputs)
        activations = self.get_activations(preprocessed_inputs, layer_names)
        assert isinstance(activations, OrderedDict)
        activations = self._unpad(activations, num_padding)
        return activations

    def translate_images(self, images, shift):
        assert type(images) == list
        temp_file_paths = []
        for image_path in images:
            fp = self.translate_image(image_path, shift)
            temp_file_paths.append(fp)
        return temp_file_paths

    def translate_image(self, image_path: str, shift: np.array) -> str:
        """Translates and saves a temporary image to temporary_fp."""
        translated_image = self.translate(cv2.imread(image_path), shift)
        temp_file_descriptor, temporary_fp = tempfile.mkstemp(suffix=".png")
        os.close(temp_file_descriptor)
        if not cv2.imwrite(temporary_fp, translated_image):
            raise Exception(f"cv2.imwrite failed: {temporary_fp}")
        return temporary_fp

    def _pad(self, batch_images, batch_size):
        num_images = len(batch_images)
        if num_images % batch_size == 0:
            return batch_images, 0
        num_padding = batch_size - (num_images % batch_size)
        padding = np.repeat(batch_images[-1:], repeats=num_padding, axis=0)
        return np.concatenate((batch_images, padding)), num_padding

    def _unpad(self, layer_activations, num_padding):
        return change_dict(layer_activations, lambda values: values[:-num_padding or None])

    def _package(self, layer_activations, stimuli_paths, number_of_trials, require_variance):
        shapes = [a.shape for a in layer_activations.values()]
        self._logger.debug(f"Activations shapes: {shapes}")
        self._logger.debug("Packaging individual layers")
        layer_assemblies = [self._package_layer(single_layer_activations,
                                                layer=layer,
                                                stimuli_paths=stimuli_paths,
                                                number_of_trials=number_of_trials,
                                                require_variance=require_variance) for
                            layer, single_layer_activations in tqdm(layer_activations.items(), desc='layer packaging')]
        # merge manually instead of using merge_data_arrays since `xarray.merge` is very slow with these large arrays
        # complication: (non)neuroid_coords are taken from the structure of layer_assemblies[0] i.e. the 1st assembly;
        # using these names/keys for all assemblies results in KeyError if the first layer contains flatten_coord_names
        # (see _package_layer) not present in later layers, e.g. first layer = conv, later layer = transformer layer
        self._logger.debug(f"Merging {len(layer_assemblies)} layer assemblies")
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
            for coord, dims, values in walk_coords(layer_assembly):
                if set(dims) == {'neuroid'}:
                    continue
                assert (values == nonneuroid_coords[coord][1]).all()

        neuroid_coords = {coord: (dims_values[0], dims_values[1])  # re-package as tuple instead of list for xarray
                          for coord, dims_values in neuroid_coords.items()}
        model_assembly = type(layer_assemblies[0])(model_assembly, coords={**nonneuroid_coords, **neuroid_coords},
                                                   dims=layer_assemblies[0].dims)
        return model_assembly

    def _package_layer(self, layer_activations, layer, stimuli_paths, number_of_trials=1, require_variance=False):
        # activation shape is larger if variance in responses is required from the model by a factor of number_of_trials
        if require_variance:
            runs_per_image = number_of_trials
        else:
            runs_per_image = 1
        assert layer_activations.shape[0] == len(stimuli_paths) * runs_per_image
        stimuli_paths = np.repeat(stimuli_paths, runs_per_image)
        activations, flatten_indices = flatten(layer_activations, return_index=True)  # collapse for single neuroid dim
        flatten_coord_names = None
        if flatten_indices.shape[1] == 1:  # fully connected, e.g. classifier
            # see comment in _package for an explanation why we cannot simply have 'channel' for the FC layer
            flatten_coord_names = ['channel', 'channel_x', 'channel_y']
        elif flatten_indices.shape[1] == 2:  # Transformer, e.g. ViT
            flatten_coord_names = ['channel', 'embedding']
        elif flatten_indices.shape[1] == 3:  # 2DConv, e.g. resnet
            flatten_coord_names = ['channel', 'channel_x', 'channel_y']
        elif flatten_indices.shape[1] == 4:  # temporal sliding window, e.g. omnivron
            flatten_coord_names = ['channel_temporal', 'channel_x', 'channel_y', 'channel']
        else:
            # we still package the activations, but are unable to provide channel information
            self._logger.debug(f"Unknown layer activations shape {layer_activations.shape}, not inferring channels")

        # build assembly
        if require_variance:
            coords = self.build_microsaccade_coords(activations, layer, stimuli_paths)
        else:
            coords = {'stimulus_path': ('presentation', stimuli_paths),
                      'stimulus_path2': ('presentation', stimuli_paths),  # to avoid DataAssembly dim collapse
                      'neuroid_num': ('neuroid', list(range(activations.shape[1]))),
                      'model': ('neuroid', [self.identifier] * activations.shape[1]),
                      'layer': ('neuroid', [layer] * activations.shape[1]),
                      }

        if flatten_coord_names:
            flatten_coords = {flatten_coord_names[i]: [sample_index[i] if i < flatten_indices.shape[1] else np.nan
                                                       for sample_index in flatten_indices]
                              for i in range(len(flatten_coord_names))}
            coords = {**coords, **{coord: ('neuroid', values) for coord, values in flatten_coords.items()}}
        layer_assembly = NeuroidAssembly(activations, coords=coords, dims=['presentation', 'neuroid'])
        neuroid_id = [".".join([f"{value}" for value in values]) for values in zip(*[
            layer_assembly[coord].values for coord in ['model', 'layer', 'neuroid_num']])]
        layer_assembly['neuroid_id'] = 'neuroid', neuroid_id
        return layer_assembly

    def build_microsaccade_coords(self, activations, layer, stimuli_paths):
        coords = {
            'stimulus_path': ('presentation', stimuli_paths),
            'shift_x': ('presentation', [shift[0] for shift in self.shifts]),
            'shift_y': ('presentation', [shift[1] for shift in self.shifts]),
            'neuroid_num': ('neuroid', list(range(activations.shape[1]))),
            'model': ('neuroid', [self.identifier] * activations.shape[1]),
            'layer': ('neuroid', [layer] * activations.shape[1]),
        }
        return coords

    def insert_attrs(self, wrapper):
        wrapper.from_stimulus_set = self.from_stimulus_set
        wrapper.from_paths = self.from_paths
        wrapper.register_batch_activations_hook = self.register_batch_activations_hook
        wrapper.register_stimulus_set_hook = self.register_stimulus_set_hook

    @staticmethod
    def select_microsaccades(number_of_trials: int):
        """
        A naive function for generating microsaccade locations that span the same visual angle regardless of model
        visual angle (to keep microsaccade extent constant across models). The function returns a list of
        `number_of_trials` tuples that each contain a microsaccade location, expanding from the center of the image.
        """
        n = int(np.ceil(np.sqrt(number_of_trials)))  # upper bound on number of microsaccades needed

        # generate grid of potential microsaccades
        xv, yv = np.meshgrid(range(-n, n + 1), range(-n, n + 1))
        coords = np.vstack([xv.ravel(), yv.ravel()]).T

        # sort microsaccade locations: (0, 0) should be first, and e.g. (2, 3) before (-5, 6) (absolute value).
        sort_criteria = np.maximum(np.abs(coords[:, 0]), np.abs(coords[:, 1])) + np.abs(coords).sum(axis=1) / 1000
        sorted_indices = np.argsort(sort_criteria)

        # select the first `number_of_trials` microsaccades from the sorted upper bound
        selected_microsaccades = [tuple(coord) for coord in coords[sorted_indices][:number_of_trials]]
        return selected_microsaccades

    @staticmethod
    def translate(image, shift):
        rows, cols, _ = image.shape
        # translation matrix
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])

        # Apply translation, filling new line(s) with line(s) closest to it(them).
        translated_image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        return translated_image


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
    assembly = assembly.reset_index('presentation')
    assembly.assign_coords(stimulus_path=('presentation', stimulus_set['stimulus_id'].values))

    assembly = assembly.rename({'stimulus_path': 'stimulus_id'})

    all_columns = []
    for column in stimulus_set.columns:
        assembly = assembly.assign_coords({column: ('presentation', stimulus_set[column].values)})
        all_columns.append(column)
    if 'stimulus_id' in all_columns:
        all_columns.remove('stimulus_id')
    if 'stimulus_path2' in all_columns:
        all_columns.remove('stimulus_path2')
    assembly = assembly.set_index(presentation=['stimulus_id', 'stimulus_path2'] + all_columns)
    return assembly


def attach_stimulus_set_meta_with_microsaccades(assembly, stimulus_set, number_of_trials, shifts):
    stimulus_paths = [str(stimulus_set.get_stimulus(stimulus_id)) for stimulus_id in stimulus_set['stimulus_id']]
    stimulus_paths = [lstrip_local(path) for path in stimulus_paths]
    assembly_paths = [lstrip_local(path) for path in assembly['stimulus_path'].values]

    replication_factor = number_of_trials
    repeated_stimulus_paths = np.repeat(stimulus_paths, replication_factor)
    assert (np.array(assembly_paths) == np.array(repeated_stimulus_paths)).all()
    repeated_stimulus_ids = np.repeat(stimulus_set['stimulus_id'].values, replication_factor)

    # repeat over the presentation dimension to accommodate multiple runs per stimulus
    repeated_assembly = xr.concat([assembly for _ in range(replication_factor)], dim='presentation')
    repeated_assembly = repeated_assembly.reset_index('presentation')
    repeated_assembly.assign_coords(stimulus_path=('presentation', repeated_stimulus_ids))
    repeated_assembly = repeated_assembly.rename({'stimulus_path': 'stimulus_id'})

    # hack to capture columns
    all_columns = []
    for column in stimulus_set.columns:
        repeated_values = np.repeat(stimulus_set[column].values, replication_factor)
        repeated_assembly = repeated_assembly.assign_coords({column: ('presentation', repeated_values)})
        all_columns.append(column)
    if 'stimulus_id' in all_columns:
        all_columns.remove('stimulus_id')
    if 'stimulus_path2' in all_columns:
        all_columns.remove('stimulus_path2')

    repeated_assembly.coords['shift_x'] = ('presentation', [shift[0] for shift in shifts])
    repeated_assembly.coords['shift_y'] = ('presentation', [shift[1] for shift in shifts])

    # Set MultiIndex
    index = ['stimulus_id', 'stimulus_path2'] + all_columns + ['shift_x', 'shift_y']
    repeated_assembly = repeated_assembly.set_index(presentation=index)

    return repeated_assembly


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
