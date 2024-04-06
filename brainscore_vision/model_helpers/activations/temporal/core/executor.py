import os
import logging

import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict
from typing import Any, Callable
from ..inputs import Stimulus

from brainscore_vision.model_helpers.utils import fullname
from joblib import Parallel, delayed


def pipeline(*funcs):
    def _pipeline(x, *others):
        for f in funcs:
            x = f(x, *others)
        return x
    return _pipeline

    
class JoblibMapper:
    def __init__(self, num_threads):
        self._num_threads = num_threads
        self._pool = Parallel(n_jobs=num_threads, verbose=False, backend="loky")

    def map(self, func, *data):
        return self._pool(delayed(func)(*x) for x in zip(*data))


class BatchExecutor:
    """Executor for batch processing of stimuli.
    
    Parameters
    ----------
        get_activations : function 
            function that takes a list of processed stimuli and a list of layers, and returns a dictionary of activations.
        preprocessing : function
            function that takes a stimulus and returns a processed stimulus.
        batch_size: int
            number of stimuli to process in each batch.
        batch_padding: bool
            whether to pad the each batch with the last stimulus to make it the same size as the specified batch size. 
            Otherwise, some batches will have size < batch_size because of the lacking of samples in that group.
        batch_grouper: function
            function that takes a stimulus and return the property based on which the stimuli can be grouped.
        max_workers: int
            number of workers for parallel processing. If None, the number of workers will be the number of cpus.

    APIs
    ----
    add_stimuli(stimuli)
        add a list of stimulus to the executor.
    clear_stimuli()
        clear the kept stimuli list.
    execute(layers)
        execute the batch processing of the kept stimuli and return the activations of the specified layers.
        For each layer, the activation of different stimuli will be stacked.
        If they have different shapes, the smaller ones will be padded with NaNs.

    Examples
    --------
    >>> executor = BatchExecutor(get_activations, preprocessing, batch_size=64, batch_grouper=lambda s: s.size)
    >>> executor.register_before_hook(lambda v: v.set_size(180, 120))
    >>> executor.register_after_hook(lambda val, l, s: val if l == "layer1" else val.mean(0))
    >>> executor.add_stimuli(stimulus_list_1)
    >>> executor.add_stimuli(stimulus_list_2)
    >>> layer_activations = executor.execute(layers)  # here the actual batching and processing happen
    """


    def __init__(self, get_activations, preprocessing, batch_size, batch_padding, 
                 batch_grouper=None, max_workers=None):
        self.stimuli = []
        self.get_activations = get_activations
        self.batch_size = batch_size
        self.batch_padding = batch_padding
        self.batch_grouper = batch_grouper
        self.preprocess = preprocessing
        self.max_workers = max_workers

        self._logger = logging.getLogger(fullname(self))

        # Pool for I/O intensive ops
        # torch suggest using the number of cpus as the number of threads, but os.cpu_count() returns the number of threads
        num_threads = max(min(int(os.cpu_count() / 1.5), self.batch_size), 1)  
        if self.max_workers is not None:
            num_threads = min(self.max_workers, num_threads)
        self._logger.info(f"Using {num_threads} threads for parallel processing.")
        self._mapper = JoblibMapper(num_threads)

        # processing hooks
        self.before_hooks = []
        self.after_hooks = []

    def get_batches(self, data, batch_size, grouper=None, padding=False):
        N = len(data)

        if grouper is None:
            sorted_data = np.array(data, dtype='object')
            inverse_indices = np.arange(N)
            sorted_properties = [0] * N
        else:
            properties = np.array([hash(grouper(d)) for d in data])
            sorted_indices = np.argsort(properties)
            inverse_indices = np.argsort(sorted_indices)  # inverse transform
            sorted_properties = properties[sorted_indices]
            sorted_data = np.array(data, dtype='object')[sorted_indices]
        inverse_indices = list(inverse_indices)

        index = 0
        all_batches = []
        all_indices = []
        indices = []
        masks = []
        while index < N:
            property_anchor = sorted_properties[index]
            batch = []
            while index < N and len(batch) < batch_size and sorted_properties[index] == property_anchor:
                batch.append(sorted_data[index])
                index += 1
            
            batch_indices = inverse_indices[index-len(batch):index]

            if padding:
                num_padding = batch_size - len(batch)
                if num_padding:
                    batch_padding = [batch[-1]] * num_padding
                    batch += batch_padding
            else:
                num_padding = 0
            masks.append([True] * (len(batch)-num_padding) + [False] * num_padding) 

            all_batches.append(batch)
            all_indices.append(batch_indices)
            indices.extend(batch_indices)
        return indices, masks, all_batches

    def register_before_hook(self, hook: Callable[[Stimulus], Stimulus], index=None):
        # hook: Stimulus -> Stimulus
        if index is None: index = len(self.before_hooks)
        self.before_hooks.insert(index, hook)

    def register_after_hook(self, hook: Callable[[Any, str, Stimulus], Any], index=None):
        # hook: value, layer_name, Stimulus -> value
        if index is None: index = len(self.after_hooks)
        self.after_hooks.insert(index, hook)
    
    def add_stimuli(self, stimuli):
        self.stimuli.extend(stimuli)

    def clear_stimuli(self):
        self.stimuli = []

    def execute(self, layers):
        indices, masks, batches = self.get_batches(self.stimuli, self.batch_size, 
                                                    grouper=self.batch_grouper,
                                                    padding=self.batch_padding)
        
        before_pipe = pipeline(*self.before_hooks)
        after_pipe = pipeline(*self.after_hooks)
        
        layer_activations = OrderedDict()
        for mask, batch in tqdm(zip(masks, batches), desc="activations", total=len(batches)):
            batch = [before_pipe(stimulus) for stimulus in batch]
            model_inputs = self._mapper.map(self.preprocess, batch)
            batch_activations = self.get_activations(model_inputs, layers)
            assert isinstance(batch_activations, OrderedDict)
            for layer, activations in batch_activations.items():
                results = [after_pipe(arr, layer, stimulus) 
                               for not_pad, arr, stimulus in zip(mask, activations, batch) 
                               if not_pad]
                layer_activations.setdefault(layer, []).extend(results)

        for layer, activations in layer_activations.items():
            layer_activations[layer] = [activations[i] for i in indices]

        self.clear_stimuli()
        return layer_activations