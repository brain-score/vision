import os
import time
import logging

from collections import OrderedDict
import numpy as np
from tqdm.auto import tqdm

from brainscore_vision.model_helpers.utils import fullname
from ..utils import parallelize


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
        dtype: np.dtype
            data type of the activations.

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
    >>> executor.add_stimuli(stimulus_list_1)
    >>> executor.add_stimuli(stimulus_list_2)
    >>> layer_activations = executor.execute(layers)  # here the actual batching and processing happen
    """


    def __init__(self, get_activations, preprocessing, batch_size, batch_padding, batch_grouper=None, dtype=np.float16):
        self.stimuli = []
        self.get_activations = get_activations
        self.batch_size = batch_size
        self.batch_padding = batch_padding
        self.batch_grouper = batch_grouper
        self.preprocess = preprocessing
        self.dtype = dtype

        self._logger = logging.getLogger(fullname(self))

        # Pool for I/O intensive ops

    @staticmethod
    def get_batches(data, batch_size, grouper=None, padding=False, n_jobs=1):
        N = len(data)

        if grouper is None:
            sorted_data = np.array(data, dtype='object')
            sorted_indices = np.arange(N)
            sorted_properties = [0] * N
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

        def list_collate(batch):
            return [item for item in batch if item is not None]
        
        # torch suggest using the number of cpus as the number of threads, but os.cpu_count() returns the number of threads
        num_threads = min(os.cpu_count() // 2, self.batch_size)  
        loader = DataLoader(_data(), batch_size=global_batch_size, shuffle=False, 
                            collate_fn=list_collate, num_workers=num_threads)
        return loader, indices, mask
  
    def execute(self, layers):
        loader, indices, mask = self._make_dataloader(self.stimuli)
        layer_activations = OrderedDict()
        ta = time.time()
        for batch in tqdm(loader, desc="activations"):
            tb = time.time()
            self._logger.debug(f"Time to load batch: {tb - ta:.2f}s")
            batch_activations = self.get_activations(batch, layers)
            ta = time.time()
            self._logger.debug(f"Time to process batch: {ta - tb:.2f}s")
            assert isinstance(batch_activations, OrderedDict)
            for layer, activations in batch_activations.items():
                for activation in activations:
                    layer_activations.setdefault(layer, []).append(activation.astype(self.dtype))

        for layer, activations in layer_activations.items():
            layer_activations[layer] = [activations[i] for i, not_padding in zip(indices, mask) if not_padding]
            layer_activations[layer] = stack_with_nan_padding(layer_activations[layer], axis=0, dtype=self.dtype)

        self.clear_stimuli()
        return layer_activations


def stack_with_nan_padding(arr_list, axis=0, dtype=np.float16):
    # Get shapes of all arrays
    shapes = [np.array(arr.shape) for arr in arr_list]
    max_shape = np.max(shapes, axis=0)

    # Allocate concatenated array with NaN padding
    result = np.full(np.concatenate(([len(arr_list)], max_shape)), np.nan, dtype=dtype)

    # Fill in individual arrays
    for i, arr in enumerate(arr_list):
        slices = tuple(slice(0, s) for s in arr.shape)
        result[i][slices] = arr

    return result