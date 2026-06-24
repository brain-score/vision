import os
import logging

import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict
from typing import Any, Callable, Dict, Hashable, List
from ..inputs import Stimulus

from brainscore_vision.model_helpers.utils import fullname
from joblib import Parallel, delayed


# a utility to apply a list of functions to inputs sequentially but only iterate over the first input
def _pipeline(*funcs):
    def _func(x, *others):
        for f in funcs:
            x = f(x, *others)
        return x
    return _func

    
# a mapper that execute a fxunction in parallel with joblib
class JobsMapper:
    def __init__(self, num_threads: int):
        self._num_threads = num_threads
        self._pool = Parallel(n_jobs=num_threads, verbose=False, backend="loky")
        self._failed_to_pickle_func = False

    def map(self, func, *data):
        from joblib.externals.loky.process_executor import TerminatedWorkerError, BrokenProcessPool
        if not self._failed_to_pickle_func:
            try:
                return self._pool(delayed(func)(*x) for x in zip(*data))
            except (TerminatedWorkerError, BrokenProcessPool):
                self._failed_to_pickle_func = True
        return [func(*x) for x in zip(*data)]


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


    def __init__(self, 
                get_activations : Callable[[List[Any]], Dict[str, np.array]], 
                preprocessing : Callable[[List[Stimulus]], Any], 
                batch_size : int, 
                batch_padding : bool = False, 
                batch_grouper : Callable[[Stimulus], Hashable] = None, 
                max_workers : int = None
            ):
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
        self._mapper = JobsMapper(num_threads)

        # processing hooks
        self.before_hooks = []
        self.after_hooks = []

    def _get_batches(
            self, 
            data, 
            batch_size : int, 
            grouper : Callable[[Stimulus], Hashable] = None, 
            padding : bool = False
        ):
        """Group the data into batches based on the grouper.
        
        Parameters
        ----------

        data : array-like
            list of data to be grouped.
        batch_size, grouper, padding : int, function, bool, directly set by the class

        Returns
        -------
        all_indices : list of list
            indices of the source data in each batch.
        all_masks : list of list
            masks for each batch to indicate whether the datum is padding sample.
            1 for not padding, 0 for padding.
        all_batches : list of list
            list of batches.
        """

        N = len(data)

        if grouper is None:
            sorted_data = np.array(data, dtype='object')
            sorted_indices = np.arange(N)
            sorted_properties = [0] * N
        else:
            properties = np.array([hash(grouper(d)) for d in data])
            sorted_indices = np.argsort(properties)
            sorted_properties = properties[sorted_indices]
            sorted_data = np.array(data, dtype='object')[sorted_indices]
        sorted_indices = list(sorted_indices)

        index = 0
        all_batches = []
        all_indices = []
        all_masks = []
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
            else:
                num_padding = 0
            all_masks.append([True] * (len(batch)-num_padding) + [False] * num_padding) 
            all_batches.append(batch)
            all_indices.append(batch_indices + [-1] * num_padding)
        return all_indices, all_masks, all_batches

    def _register_hook(self, name, hook, hook_group, index=None):
        if index is None: index = len(hook_group)
        hook_group.insert(index, (name, hook))

    def _remove_hook(self, name, hook_group):
        hook_group[:] = [(n, h) for n, h in hook_group if n != name]

    def register_before_hook(self, name: str, hook: Callable[[Stimulus], Stimulus], index=None):
        # hook: Stimulus -> Stimulus
        self._register_hook(name, hook, self.before_hooks, index)

    def register_after_hook(self, name: str, hook: Callable[[Any, str, Stimulus], Any], index=None):
        # hook: value, layer_name, Stimulus -> value
        self._register_hook(name, hook, self.after_hooks, index)

    def remove_before_hook(self, name: str):
        self._remove_hook(name, self.before_hooks)

    def remove_after_hook(self, name: str):
        self._remove_hook(name, self.after_hooks)

    def add_stimuli(self, stimuli):
        self.stimuli.extend(stimuli)

    def clear_stimuli(self):
        self.stimuli = []

    def execute(self, layers):
        full_indices = []
        full_activations = OrderedDict()
        for layer_activations, indices in self.execute_batch(layers):
            full_indices.extend(indices)
            for layer_activation in layer_activations:
                for layer, activations in layer_activation.items():
                    full_activations.setdefault(layer, []).append(activations)
        
        for layer, activations in full_activations.items():
            full_activations[layer] = [activations[i] for i in full_indices]

        return full_activations

    def execute_batch(self, layers):
        all_indices, all_masks, batches = self._get_batches(self.stimuli, self.batch_size, 
                                                    grouper=self.batch_grouper,
                                                    padding=self.batch_padding)

        before_pipe = _pipeline(*[hook for name, hook in self.before_hooks])
        after_pipe = _pipeline(*[hook for name, hook in self.after_hooks])

        # avoid keeping the whole batch in memory
        def run(batch, mask, indices):
            batch = [before_pipe(stimulus) for stimulus in batch]
            model_inputs = self._mapper.map(self.preprocess, batch)
            batch_activations = self.get_activations(model_inputs, layers)
            assert isinstance(batch_activations, OrderedDict)
            for i, (layer, activations) in enumerate(batch_activations.items()):
                results = [after_pipe(arr, layer, stimulus) 
                            for not_pad, arr, stimulus in zip(mask, activations, batch) 
                            if not_pad]
                if i == 0:
                    layer_activations = [OrderedDict() for _ in range(len(results))]
                
                for j, result in enumerate(results):
                    layer_activations[j][layer] = result
            return layer_activations, indices

        for indices, mask, batch in tqdm(zip(all_indices, all_masks, batches), desc="activations", total=len(batches)):
            yield run(batch, mask, indices)
            
        self.clear_stimuli()