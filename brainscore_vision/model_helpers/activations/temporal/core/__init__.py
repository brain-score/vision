import copy
import os

import functools
import logging
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
import time

import numpy as np
from tqdm.auto import tqdm

from brainio.assemblies import NeuroidAssembly, walk_coords
from brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.utils import fullname
from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper
from result_caching import store_xarray
from ..inputs import Video
from ..utils import parallelize, assembly_align_to_fps
from . import time_aligner


channel_name_mapping = {
    "T": "channel_temporal",
    "C": "channel",
    "H": "channel_y",
    "W": "channel_x",
    "K": "channel_token"
}


class Defaults:
    batch_size = 8


class BatchInferencer:
    def __init__(self, get_activations, preprocessing, batch_size, batch_padding, batch_grouper=None, dtype=np.float16):
        self.stimuli = []
        self.get_activations = get_activations
        self.batch_size = batch_size
        self.batch_padding = batch_padding
        self.batch_grouper = batch_grouper
        self.preprocess = preprocessing
        self.dtype = dtype

        # Pool for I/O intensive ops

    @staticmethod
    def get_batches(data, batch_size, grouper=None, padding=False, n_jobs=1):
        N = len(data)

        if grouper is None:
            sorted_data = np.array(data, dtype='object')
            sorted_indices = np.arange(N)
        else:
            properties = parallelize(grouper, data, n_jobs=n_jobs)
            properties = np.array([hash(p) for p in properties])
            sorted_indices = np.argsort(properties)
            sorted_properties = properties[sorted_indices]
            sorted_data = np.array(data, dtype='object')[sorted_indices]
        sorted_indices = list(sorted_indices)

        index = 0
        all_batches = []
        all_indices = []
        indices = []
        mask = []
        while index < N:
            property_anchor = sorted_properties[index]
            batch = []
            while index < N and len(batch) < batch_size and sorted_properties[index] == property_anchor:
                batch.append(sorted_data[index])
                index += 1
            
            batch_indices = sorted_indices[index-len(batch):index]

            if padding:
                num_padding = batch_size - len(batch)
                if num_padding:
                    batch_padding = [batch[-1]] * num_padding
                    batch += batch_padding
                    indices_padding = [None] * num_padding
                    batch_indices += indices_padding
            else:
                num_padding = 0
            mask += [True] * (len(batch)-num_padding) + [False] * num_padding

            all_batches.append(batch)
            all_indices.append(batch_indices)
            indices.extend(batch_indices)
        return indices, mask, all_batches

    def add_stimuli(self, stimuli):
        self.stimuli.extend(stimuli)

    def clear_stimuli(self):
        self.stimuli = []
        
    def _make_dataloader(self, stimuli):
        from torch.utils.data import Dataset, DataLoader

        indices, mask, batches = self.get_batches(stimuli, self.batch_size, 
                                                    grouper=self.batch_grouper,
                                                    padding=self.batch_padding)
        global_batch_size = self.batch_size
        preprocessing = self.preprocess

        class _data(Dataset):
            def __init__(self):
                self.batch = []
                for batch in batches:
                    if len(batch) < global_batch_size:
                        num_padding = global_batch_size - len(batch)
                    else:
                        num_padding = 0
                    self.batch.extend(batch + [None] * num_padding)
                        
            def __len__(self):
                return len(self.batch)
            
            def __getitem__(self, idx):
                item = self.batch[idx]
                if item is not None:
                    return preprocessing(item)
                else:
                    return None

        def my_collate(batch):
            return [item for item in batch if item is not None]
        
        # torch suggest using the number of cpus as the number of threads, but os.cpu_count() returns the number of threads
        num_threads = os.cpu_count() // 2  
        loader = DataLoader(_data(), batch_size=global_batch_size, shuffle=False, 
                            collate_fn=my_collate, num_workers=num_threads)
        return loader, indices, mask

        
    def inference(self, layers):
        loader, indices, mask = self._make_dataloader(self.stimuli)
        layer_activations = OrderedDict()
        for batch in tqdm(loader, desc="activations"):
            batch_activations = self.get_activations(batch, layers)
            assert isinstance(batch_activations, OrderedDict)
            for layer, activations in batch_activations.items():
                for activation in activations:
                    layer_activations.setdefault(layer, []).append(activation.astype(self.dtype))

        for layer, activations in layer_activations.items():
            layer_activations[layer] = [activations[i] for i, not_padding in zip(indices, mask) if not_padding]

        self.clear_stimuli()
        return layer_activations


class TemporalExtractorHelper(ActivationsExtractorHelper):
    def __init__(
            self, 
            get_activations, 
            preprocessing,
            spec, 
            identifier=False,
            batch_size=Defaults.batch_size, 
            batch_padding=False,
            batch_grouper=lambda video: (video.duration, video.fps, video.frame_size),
            causal_inference=False,
            time_alignment="evenly_spaced",
            max_temporal_context=2000, # ms
            dtype=np.float16,
        ):
        self._logger = logging.getLogger(fullname(self))

        self.identifier = identifier
        self._get_activations = get_activations
        self.spec = spec.copy()
        self.input_spec = spec['input']
        self.activation_spec = spec['activation']

        # Batching
        self._batch_inferencer = BatchInferencer(get_activations, preprocessing, batch_size, 
                                                 batch_padding, batch_grouper, dtype)

        # Temporal specific
        self.fps = self.input_spec['fps']
        self.input_process = Video.from_path
        self.time_alignment = time_alignment
        self.causal_inference = causal_inference
        self.max_temporal_context = max_temporal_context

        # efficient storage
        self.dtype = dtype

        self._stimulus_set_hooks = {}
        self._batch_activations_hooks = {}

    def _compute_temporal_context(self, input_spec):
        input_spec = input_spec.copy()
        context_specified = lambda: 'duration' in input_spec or 'num_frames' in input_spec
        if self.max_temporal_context:
            if not context_specified():
                input_spec['duration'] = self.max_temporal_context
        
        if context_specified():
            if 'num_frames' in input_spec:  # prioritize num_frames
                context = input_spec['num_frames'] * 1000 / self.fps
            else:
                context = input_spec['duration']
        else:
            context = None

        return context

    def get_activations(self, inputs, layers):
        if self.causal_inference:
            interval = 1000 / self.fps
            num_clips = []
            for inp in inputs:
                duration = inp.duration
                videos = []
                for time_end in np.arange(duration, 0, -interval)[::-1]:
                    # see if the model only receive limited context
                    context = self._compute_temporal_context(self.input_spec)
                    time_start = time_end - context if context else 0
                    videos.append(inp.set_window(time_start, time_end))

                self._batch_inferencer.add_stimuli(videos)
                num_clips.append(len(videos))

            activations = self._batch_inferencer.inference(layers)
            layer_activations = OrderedDict()
            for layer in layers:
                activation_dims = self.activation_spec[layer]
                clip_start = 0
                for num_clip in num_clips:
                    video_activations = activations[layer][clip_start:clip_start+num_clip]  # clips for this video
                    if 'T' in activation_dims:
                        time_index = activation_dims.index('T')
                        video_activations = [a.take([-1], axis=time_index) for a in video_activations]
                    else:
                        # if the activation dimension does not contain T,
                        # make it the first dimension, as [T, ...]
                        time_index = 0
                        video_activations = [a[None, ...] for a in video_activations]
                    layer_activations.setdefault(layer, []).append(np.concatenate(video_activations, axis=time_index))
                    clip_start += num_clip
            return layer_activations
        else:
            self._batch_inferencer.add_stimuli(inputs)
            return self._batch_inferencer.inference(layers)
    
    def _from_paths(self, layers, stimuli_paths):
        if len(layers) == 0:
            raise ValueError("No layers passed to retrieve activations from")
        self._logger.info(f'Running stimuli. FPS set to {self.fps}...')
        stimuli = [self.input_process(path).set_fps(self.fps) for path in stimuli_paths]
        layer_activations = self.get_activations(stimuli, layers=layers)
        self._logger.info('Packaging into assembly')
        return self._package(layer_activations, stimuli_paths, stimuli)

    def _package(self, layer_activations, stimuli_paths, stimuli):
        self._logger.debug("Packaging individual layers")
        layer_assemblies = [self._package_layer(single_layer_activations, layer=layer, stimuli_paths=stimuli_paths, stimuli=stimuli) for
                            layer, single_layer_activations in tqdm(layer_activations.items(), desc='layer packaging')]
        # align all layer_assemblies to the fps
        layer_assemblies = [assembly_align_to_fps(assembly, self.fps) for assembly in layer_assemblies]
        layer_assemblies = [asm.transpose("stimulus_path", "time_bin", "neuroid") for asm in layer_assemblies]

        # merge manually instead of using merge_data_arrays since `xarray.merge` is very slow with these large arrays
        # complication: (non)neuroid_coords are taken from the structure of layer_assemblies[0] i.e. the 1st assembly;
        # using these names/keys for all assemblies results in KeyError if the first layer contains flatten_coord_names
        # (see _package_layer) not present in later layers, e.g. first layer = conv, later layer = transformer layer
        self._logger.debug(f"Merging {len(layer_assemblies)} layer assemblies")
        model_assembly = np.concatenate([a.values for a in layer_assemblies],
                                        axis=layer_assemblies[0].dims.index('neuroid'))
        
        nonneuroid_coords = {coord: (dims, values) for coord, dims, values in walk_coords(layer_assemblies[0])
                             if set(dims) != {'neuroid'}}
        neuroid_coords = [(coord, dims) for layer_assembly in layer_assemblies for coord, dims, values in walk_coords(layer_assembly)
                             if set(dims) == {'neuroid'}]
        neuroid_coord_names = set(neuroid_coords)
        neuroid_coords = {}

        for layer_assembly in layer_assemblies:
            for coord, _ in neuroid_coord_names:
                try:
                    coord_values = layer_assembly[coord].values
                except KeyError:
                    coord_values = np.full(layer_assembly.sizes['neuroid'], -1, dtype=int)
                neuroid_coords.setdefault(coord, []).append(coord_values)

            assert layer_assemblies[0].dims == layer_assembly.dims
            for dim in set(layer_assembly.dims) - {'neuroid'}:
                for coord, _, _ in walk_coords(layer_assembly[dim]):
                    assert (layer_assembly[coord].values == layer_assemblies[0][coord].values).all()

        for coord, dims in neuroid_coord_names:
            neuroid_coords[coord] = (dims, np.concatenate(neuroid_coords[coord]))

        model_assembly = type(layer_assemblies[0])(model_assembly, coords={**nonneuroid_coords, **neuroid_coords},
                                                   dims=layer_assemblies[0].dims)
        return model_assembly

    def _package_layer(self, layer_activations, layer, stimuli_paths, stimuli):
        assert len(layer_activations) == len(stimuli_paths)
        if layer == "logits": layer = list(self.activation_spec.keys())[-1]
        activation_dims = self.activation_spec[layer]
        activations = layer_activations

        if self.causal_inference and "T" not in activation_dims:
            activation_dims = "T" + activation_dims
        
        if self.time_alignment == "ignore_time" or "T" not in activation_dims:
            activations = [a[None, ...] for a in activations]
            flatten_dims = activation_dims
            aligner = time_aligner.ignore_time 
        else:
            # swap the T dim to the second
            T_axis = activation_dims.index("T")
            activations = [np.swapaxes(a, 0, T_axis) for a in activations]
            flatten_dims = activation_dims.replace("T", activation_dims[0])
            flatten_dims = flatten_dims[1:]
            aligner = getattr(time_aligner, self.time_alignment)

        activations = make_full_matrix(activations)
        num_t = activations.shape[1]
        longest_stimulus = stimuli[np.argmax(np.array([stimulus.duration for stimulus in stimuli]))]

        # the length of time dimension should correspond to the longest stimulus now
        time_bin_starts, time_bin_ends = aligner(num_t, longest_stimulus)

        activations, flatten_indices = flatten(activations, return_index=True, from_index=2)  # collapse for single neuroid dim
        flatten_coord_names = [channel_name_mapping[dim] for dim in flatten_dims]
    
        # build assembly
        coords = {'stimulus_path': stimuli_paths,
                  'time_bin_start': ('time_bin', time_bin_starts),
                  'time_bin_end': ('time_bin', time_bin_ends),
                  'neuroid_num': ('neuroid', list(range(activations.shape[2]))),
                  'model': ('neuroid', [self.identifier] * activations.shape[2]),
                  'layer': ('neuroid', [layer] * activations.shape[2]),
                  }
        flatten_coords = {flatten_coord_names[i]: [sample_index[i] if i < flatten_indices.shape[1] else np.nan
                                                    for sample_index in flatten_indices]
                            for i in range(len(flatten_coord_names))}
        coords = {**coords, **{coord: ('neuroid', values) for coord, values in flatten_coords.items()}}
        layer_assembly = NeuroidAssembly(activations, coords=coords, dims=['stimulus_path', 'time_bin', 'neuroid'])
        neuroid_id = [".".join([f"{value}" for value in values]) for values in zip(*[
            layer_assembly[coord].values for coord in ['model', 'layer', 'neuroid_num']])]
        layer_assembly['neuroid_id'] = 'neuroid', neuroid_id

        return layer_assembly


def make_full_matrix(ragged_array):
    lens = []
    ragged_matrix = []
    for arr in ragged_array:
        len_ = arr.shape[0]
        lens.append(len_)
        ragged_matrix.append(arr)
    longest = max(lens)
    dtype = ragged_matrix[0].dtype
    full_matrix = np.full((len(ragged_matrix), longest, *arr.shape[1:]), dtype=dtype, fill_value=np.nan)
    for i, row in enumerate(ragged_matrix):
        full_matrix[i, :len(row)] = row
    return full_matrix


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
