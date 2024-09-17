import os
import logging
import random

import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict
from typing import Any, Callable, Dict, Hashable, List
from ..inputs import Stimulus

from brainscore_vision.model_helpers.utils import fullname
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms  # Import torchvision transforms
from jepa.src.models.attentive_pooler import AttentiveClassifier  # Ensure this import path is correct
from brainio.assemblies import NeuroidAssembly

# a utility to apply a list of functions to inputs sequentially but only iterate over the first input
def _pipeline(*funcs):
    def _func(x, *others):
        for f in funcs:
            x = f(x, *others)
        return x
    return _func

    
# a mapper that execute a function in parallel with joblib
class JoblibMapper:
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
        self._mapper = JoblibMapper(num_threads)

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
        indices : list
            indices of the source data after sorting.
        masks : list of list
            masks for each batch to indicate whether the datum is padding sample.
        all_batches : list of list
            list of batches.
        """

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
        indices, masks, batches = self._get_batches(self.stimuli, self.batch_size, 
                                                    grouper=self.batch_grouper,
                                                    padding=self.batch_padding)
        
        before_pipe = _pipeline(*self.before_hooks)
        after_pipe = _pipeline(*self.after_hooks)
        
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

# Define the ReadoutModel class
class ReadoutModel(nn.Module):
    def __init__(self, embed_dim=252, num_classes=1):
        super(ReadoutModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.linear_layer = None
        self.attentive_pooler = AttentiveClassifier(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x, mode=None):
        x = x.float()
        x = x.view(x.shape[0], x.shape[1], -1)  # Flatten keeping the last dimension
        N, T, D = x.shape
        if self.linear_layer is None:
            self.linear_layer = nn.Sequential(
                nn.Linear(D, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
            self.linear_layer = self.linear_layer.to(self.device)
        x = self.linear_layer(x.flatten(0, 1))
        x = x.view(N, T, self.embed_dim)
        x = self.attentive_pooler(x)
        if self.num_classes == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=-1)

# Define the WarmupScheduler class
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, initial_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

# Define the OnlineExecutor class
class OnlineExecutor(BatchExecutor):
    """Executor for online processing using a readout model with data augmentation, warmup, validation, and early stopping.
    
    Parameters
    ----------
    get_activations : function 
        Function that takes a list of processed stimuli and a list of layers, and returns a dictionary of activations.
    preprocessing : function
        Function that takes a stimulus and returns a processed stimulus.
    batch_size: int
        Number of stimuli to process in each batch.
    batch_padding: bool
        Whether to pad the batch with the last stimulus to make it the same size as the specified batch size.
    batch_grouper: function
        Function that takes a stimulus and returns the property based on which the stimuli can be grouped.
    max_workers: int
        Number of workers for parallel processing. If None, the number of workers will be the number of CPUs.
    readout_model_params: dict
        Parameters for initializing the ReadoutModel.
    augmentation_function: Callable
        Function to apply data augmentation to the batch before processing.
    n_epochs: int
        Number of epochs to train the readout model.
    lr: float
        Learning rate for training the readout model.
    """

    def __init__(self, 
                 get_activations: Callable[[List[Any]], Dict[str, np.array]], 
                 preprocessing: Callable[[List[Stimulus]], Any], 
                 batch_size: int, 
                 batch_padding: bool = False, 
                 batch_grouper: Callable[[Stimulus], Hashable] = None, 
                 max_workers: int = None,
                 augmentation_function: Callable[[torch.Tensor], torch.Tensor] = None,
                 n_epochs: int = 1000,
                 lr: float = 1e-3,
                 num_classes: int = 1):
        super().__init__(get_activations, preprocessing, batch_size, batch_padding, batch_grouper, max_workers)
        # Initialize the readout model with the given parameters
        self.readout_model = ReadoutModel(num_classes=num_classes)
        self.n_epochs = n_epochs
        self.lr = lr

        # Data augmentation function
        self.augmentation_function = self.default_augmentation_function

        # Loss function and optimizer for training the readout model
        self.num_classes = num_classes
        self.criterion = nn.BCELoss() if self.num_classes == 1 else nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.readout_model.parameters(), lr=self.lr)

        # Early stopping parameters
        self.patience = 300
        self.best_loss = float('inf')
        self.no_improvement_count = 0

    def default_augmentation_function(self, video_batch):
        """
        Default data augmentation function for video inputs using PyTorch transforms.
        
        Parameters
        ----------
        video_batch : torch.Tensor
            A batch of videos to augment.
        
        Returns
        -------
        augmented_batch : torch.Tensor
            Augmented batch of videos.
        """
        # Define the augmentation transformations
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=(video_batch[0].shape[-2], video_batch[0].shape[-1]), scale=(0.95, 1.0)),
        ])
        
        augmented_batch = []
        for video in video_batch:
            augmented_video = torch.stack([transform(frame) for frame in video])  # Apply augmentation frame-by-frame
            augmented_batch.append(augmented_video)
        return augmented_batch

    def _get_batches_trainer(
            self, 
            data, 
            batch_size: int, 
            padding: bool = False
        ):
        """Group the data into balanced batches based on the grouper,
        with random reuse of the minority class samples to maintain balance.
        
        Parameters
        ----------
        data : array-like
            List of data to be grouped, where each item is a tuple (video, label).
        batch_size, grouper, padding : int, function, bool, directly set by the class.
    
        Returns
        -------
        indices : list
            Indices of the source data after sorting.
        masks : list of list
            Masks for each batch to indicate whether the datum is a padding sample.
        all_batches : list of list
            List of batches, where each batch is a list of videos.
        all_labels : list of list
            List of batches, where each batch is a list of labels corresponding to the videos.
        """
        
        N = len(data)
        
        # Separate videos and labels for easier processing
        videos, labels, _ = zip(*data)  # Unzip the list of tuples into separate lists
        videos = np.array(videos, dtype='object')  # Convert to numpy array
        labels = np.array(labels, dtype='object')
        
        # Separate positive and negative samples
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        pos_videos, pos_labels = videos[pos_indices], labels[pos_indices]
        neg_videos, neg_labels = videos[neg_indices], labels[neg_indices]
        
        sorted_pos_indices = np.arange(len(pos_videos))  # Default sorted indices
        sorted_neg_indices = np.arange(len(neg_videos))  # Default sorted indices
        
        # Sorted videos and labels based on properties
        sorted_pos_videos = pos_videos[sorted_pos_indices]
        sorted_pos_labels = pos_labels[sorted_pos_indices]
        sorted_neg_videos = neg_videos[sorted_neg_indices]
        sorted_neg_labels = neg_labels[sorted_neg_indices]
    
        index_pos, index_neg = 0, 0
        all_batches = []
        all_labels = []
        all_indices = []
        indices = []
        masks = []
    
        while index_pos < len(sorted_pos_videos) or index_neg < len(sorted_neg_videos):
            batch_videos = []
            batch_labels = []
            batch_indices = []
    
            # Fill the batch with an equal number of positives and negatives
            while len(batch_videos) < batch_size:
                if index_pos < len(sorted_pos_videos) and (len(batch_videos) < batch_size / 2):
                    batch_videos.append(sorted_pos_videos[index_pos])
                    batch_labels.append(sorted_pos_labels[index_pos])
                    batch_indices.append(pos_indices[sorted_pos_indices[index_pos]])
                    index_pos += 1
                elif index_neg < len(sorted_neg_videos) and (len(batch_videos) < batch_size):
                    batch_videos.append(sorted_neg_videos[index_neg])
                    batch_labels.append(sorted_neg_labels[index_neg])
                    batch_indices.append(neg_indices[sorted_neg_indices[index_neg]])
                    index_neg += 1
                else:
                    break
    
            # If we run out of one class, randomly resample from the minority class
            while len(batch_videos) < batch_size:
                if len(batch_videos) < batch_size / 2:
                    # Resample from the positives if we're short on positives
                    random_index = random.choice(sorted_pos_indices)
                    batch_videos.append(pos_videos[random_index])
                    batch_labels.append(1)
                    batch_indices.append(pos_indices[random_index])
                else:
                    # Resample from the negatives if we're short on negatives
                    random_index = random.choice(sorted_neg_indices)
                    batch_videos.append(neg_videos[random_index])
                    batch_labels.append(0)
                    batch_indices.append(neg_indices[random_index])
    
            if padding:
                num_padding = batch_size - len(batch_videos)
                if num_padding:
                    padding_video = batch_videos[-1]  # Get the last video for padding
                    padding_label = batch_labels[-1]  # Get the last label for padding
                    batch_videos += [padding_video] * num_padding  # Add padding videos
                    batch_labels += [padding_label] * num_padding  # Add padding labels
            else:
                num_padding = 0
    
            mask = [True] * (len(batch_videos) - num_padding) + [False] * num_padding
            
            # Shuffle the combined batch
            combined = list(zip(batch_videos, batch_labels, batch_indices, mask))
            random.shuffle(combined)
            batch_videos, batch_labels, batch_indices, mask = zip(*combined)

            all_batches.append(batch_videos)
            all_labels.append(batch_labels)
            all_indices.append(batch_indices)
            indices.extend(batch_indices)
            masks.append(mask)
    
        return indices, masks, all_batches, all_labels


    def _get_batches(
        self, 
        data, 
        batch_size: int, 
        grouper: Callable[[Stimulus], Hashable] = None, 
        padding: bool = False
    ):
        """Group the data into batches based on the grouper.
    
        Parameters
        ----------
        data : array-like
            List of data to be grouped, where each item is a tuple (video, label).
        batch_size, grouper, padding : int, function, bool, directly set by the class.
    
        Returns
        -------
        indices : list
            Indices of the source data after sorting.
        masks : list of list
            Masks for each batch to indicate whether the datum is a padding sample.
        all_batches : list of list
            List of batches, where each batch is a list of videos.
        all_labels : list of list
            List of batches, where each batch is a list of labels corresponding to the videos.
        """
    
        N = len(data)
    
        # Separate videos and labels for easier processing
        videos, labels, _ = zip(*data)  # Unzip the list of tuples into separate lists
        videos = np.array(videos, dtype='object')  # Convert to numpy array
        labels = np.array(labels, dtype='object')
    
        if grouper is None:
            sorted_indices = np.arange(N)  # Default sorted indices
            sorted_properties = [0] * N  # Dummy properties since no grouping is required
        else:
            properties = np.array([hash(grouper(video)) for video in videos])  # Hash the properties
            sorted_indices = np.argsort(properties)
            sorted_properties = properties[sorted_indices]
    
        # Sort videos and labels based on properties
        sorted_videos = videos[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        inverse_indices = np.argsort(sorted_indices)  # Inverse transform for retrieving original order
        inverse_indices = list(inverse_indices)
    
        index = 0
        all_batches = []
        all_labels = []
        all_indices = []
        indices = []
        masks = []
        while index < N:
            property_anchor = sorted_properties[index]
            batch_videos = []
            batch_labels = []
            while index < N and len(batch_videos) < batch_size and sorted_properties[index] == property_anchor:
                batch_videos.append(sorted_videos[index])
                batch_labels.append(sorted_labels[index])
                index += 1
    
            batch_indices = inverse_indices[index - len(batch_videos):index]
    
            if padding:
                num_padding = batch_size - len(batch_videos)
                if num_padding:
                    padding_video = batch_videos[-1]  # Get the last video for padding
                    padding_label = batch_labels[-1]  # Get the last label for padding
                    batch_videos += [padding_video] * num_padding  # Add padding videos
                    batch_labels += [padding_label] * num_padding  # Add padding labels
            else:
                num_padding = 0
    
            masks.append([True] * (len(batch_videos) - num_padding) + [False] * num_padding)
    
            all_batches.append(batch_videos)
            all_labels.append(batch_labels)
            all_indices.append(batch_indices)
            indices.extend(batch_indices)
    
        return indices, masks, all_batches, all_labels

    def pad_tensors_to_max_length(self, tensors):
        """
        Pads a list of tensors to the maximum length along the time dimension (T).
    
        Parameters
        ----------
        tensors : list of torch.Tensor
            A list of tensors, each of shape (3, T, 224, 224) where T may vary.
    
        Returns
        -------
        padded_tensors : list of torch.Tensor
            A list of tensors, each padded to the maximum length along the time dimension.
        """
        # Determine the maximum length along the time dimension (T)
        max_length = max(tensor.size(1) for tensor in tensors)  # max(T)
    
        # Pad each tensor along the time dimension to the maximum length
        padded_tensors = []
        for tensor in tensors:
            C, T, H, W = tensor.shape
            # Calculate the amount of padding needed
            pad_length = max_length - T
            if pad_length > 0:
                # Pad tensor with zeros to make it (3, max_length, 224, 224)
                padding = (0, 0, 0, 0, 0, pad_length)  # (width, height, time)
                padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
            else:
                padded_tensor = tensor
    
            padded_tensors.append(padded_tensor)
    
        return padded_tensors

    def get_dummy_activations(self, model_inputs, layers, features_dict):
        """
        Generates a dictionary of dummy activations with zero tensors matching the shape of the model outputs.
        
        Parameters
        ----------
        model_inputs : list
            The preprocessed input stimuli.
        layers : list
            The list of layers for which activations are required.
        
        Returns
        -------
        batch_activations : OrderedDict
            A dictionary where each key corresponds to a layer and each value is a zero tensor matching 
            the shape of the model's output for that layer.
        """
        batch_activations = OrderedDict()
        for layer in layers:
            # Simulate a dummy output with zeros of the appropriate shape
            dummy_shape = (len(model_inputs), *features_dict[layer].shape[1:])  # Adjust this to the actual output shape needed
            batch_activations[layer] = np.zeros(dummy_shape)
        return batch_activations

    def execute(self, layers, train=False):
        if train:
            return self.execute_train(layers)
        return self.execute_test(layers)
        
    def execute_train(self, layers):
        seed = 42  # You can choose any integer you like
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    
        # Ensure deterministic behavior for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        indices, masks, batches, labels = self._get_batches_trainer(self.stimuli, self.batch_size, 
                                                    padding=self.batch_padding)

        before_pipe = _pipeline(*self.before_hooks)
        after_pipe = _pipeline(*self.after_hooks)

        shuffled_data = list(zip(indices, masks, batches, labels))
        random.shuffle(shuffled_data)
        indices, masks, batches, labels = zip(*shuffled_data)

        # Split data into training and validation
        validation_split = 0.1
        num_validation_batches = int(validation_split * len(batches))
        train_batches = list(zip(masks, batches, labels))[:-num_validation_batches]
        val_batches = list(zip(masks, batches, labels))[-num_validation_batches:]

        # Training loop with early stopping
        # Initialize warmup scheduler and cosine annealing scheduler
        self.total_steps = self.n_epochs * len(self.stimuli) // self.batch_size
        self.warmup_steps = int(0.05 * self.total_steps)
        self.warmup_scheduler = WarmupScheduler(self.optimizer, self.warmup_steps, self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_steps - self.warmup_steps, eta_min=0)

        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move the model to GPU and enable data parallelism if multiple GPUs are available
        self.readout_model.to(device)
        #if torch.cuda.device_count() > 1:
        #    self.readout_model = torch.nn.DataParallel(self.readout_model)

        self.best_accuracy = 0
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            total_correct = 0
            total_steps, total_samples = 0, 0
            self.readout_model.train()

            # Training Phase
            with tqdm(train_batches, desc=f"Training Epoch {epoch+1}", total=len(train_batches)) as pbar:
                for idx, (mask, batch, label) in enumerate(pbar):
                    batch = [before_pipe(stimulus) for stimulus in batch]
                    model_inputs = self._mapper.map(self.preprocess, batch)
                    model_inputs = self.pad_tensors_to_max_length(model_inputs)
                    if self.augmentation_function:
                        model_inputs = self.augmentation_function(model_inputs)  # Use the augmentation function on the batch
                    features_dict = self.get_activations(model_inputs, layers)
                    for layer, features in features_dict.items():
                        features_tensor = torch.tensor(np.stack(features)).to(device)
                        labels = torch.tensor(label).to(device)
                        
                        self.optimizer.zero_grad()
                        readout_outputs = self.readout_model(features_tensor)
                        readout_outputs = readout_outputs.squeeze()
                        loss = self.criterion(readout_outputs, labels.float())
                        loss.backward()
                        self.optimizer.step()
        
                        epoch_loss += loss.item()
        
                        # Calculate accuracy
                        predicted = (readout_outputs > 0.5).float()  # Assuming a binary classification with threshold 0.5
                        correct = (predicted == labels.float()).sum().item()
                        total_correct += correct
                        total_samples += labels.size(0)
                        total_steps += 1
        
                        # Update tqdm bar with the average accuracy
                        average_accuracy = total_correct / total_samples
                        pbar.set_postfix({'loss': epoch_loss / total_steps, 'accuracy': average_accuracy * 100})
        
                        self.warmup_scheduler.step()  # Apply warmup scheduler

            self.scheduler.step()  # Apply cosine scheduler
            
            # Validation Phase
            val_loss = 0.0
            total_correct = 0
            total_samples = 0
            self.readout_model.eval()
            
            with torch.no_grad():
                with tqdm(val_batches, desc="Validation", total=len(val_batches)) as pbar:
                    for mask, batch, label in pbar:
                        batch = [before_pipe(stimulus) for stimulus in batch]
                        model_inputs = self._mapper.map(self.preprocess, batch)
                        model_inputs = self.pad_tensors_to_max_length(model_inputs)
                        features_dict = self.get_activations(model_inputs, layers)
            
                        for layer, features in features_dict.items():
                            features_tensor = torch.tensor(np.stack(features)).to(device)
                            labels = torch.tensor(label).to(device)
                            readout_outputs = self.readout_model(features_tensor)
                            readout_outputs = readout_outputs.squeeze() 
                            # Compute loss
                            loss = self.criterion(readout_outputs, labels.float())
                            val_loss += loss.item()
            
                            # Calculate accuracy
                            predicted = (readout_outputs > 0.5).float()  # Assuming binary classification
                            correct = (predicted == labels.float()).sum().item()
                            total_correct += correct
                            total_samples += labels.size(0)
            
                            # Update tqdm bar with loss and accuracy
                            avg_val_loss = val_loss / (total_samples / labels.size(0))
                            avg_val_accuracy = total_correct / total_samples * 100
                            pbar.set_postfix({'loss': avg_val_loss, 'accuracy': avg_val_accuracy})
             
            # Compute the average loss and accuracy
            avg_val_loss = val_loss / len(val_batches)
            avg_val_accuracy = total_correct / total_samples * 100
            print(f"Epoch {epoch+1} completed with training loss: {epoch_loss/len(train_batches)}, validation loss: {avg_val_loss}, validation accuracy: {avg_val_accuracy:.2f}%")
            self._logger.info(f"Epoch {epoch+1} completed with training loss: {epoch_loss/len(train_batches)}, validation loss: {avg_val_loss}, validation accuracy: {avg_val_accuracy:.2f}%")
            
            # Early stopping check based on accuracy
            if avg_val_accuracy > self.best_accuracy:
                self.best_accuracy = avg_val_accuracy
                self.no_improvement_count = 0
                torch.save(self.readout_model.state_dict(), 'transformer_readout.pt')
                print(f"New best model saved with validation accuracy: {avg_val_accuracy:.2f}%")
                self._logger.info(f"New best model saved with validation accuracy: {avg_val_accuracy:.2f}%")
            else:
                self.no_improvement_count += 1
                if self.no_improvement_count >= self.patience:
                    self._logger.info("Early stopping triggered.")
                    break

        indices, masks, batches, labels = self._get_batches(self.stimuli, self.batch_size, 
                                                    grouper=self.batch_grouper,
                                                    padding=self.batch_padding)

        # Final execution with dummy activations
        layer_activations = OrderedDict()
        for mask, batch in tqdm(zip(masks, batches), desc="activations", total=len(batches)):
            batch = [before_pipe(stimulus) for stimulus in batch]
            model_inputs = self._mapper.map(self.preprocess, batch)

            # Get dummy activations with zero tensors that match the shape of model outputs
            batch_activations = self.get_dummy_activations(model_inputs, layers, features_dict)
            assert isinstance(batch_activations, OrderedDict)

            for layer, activations in batch_activations.items():
                results = [after_pipe(arr, layer, stimulus) 
                           for not_pad, arr, stimulus in zip(mask, activations, batch) 
                           if not_pad]
                layer_activations.setdefault(layer, []).extend(results)

        # Reorganize activations in the original order
        for layer, activations in layer_activations.items():
            layer_activations[layer] = [activations[i] for i in indices]

        self.clear_stimuli()
        return layer_activations

    def execute_test(self, layers):
        indices, masks, batches, labels = self._get_batches(self.stimuli, self.batch_size, 
                                                    grouper=self.batch_grouper,
                                                    padding=self.batch_padding)
        
        before_pipe = _pipeline(*self.before_hooks)
        after_pipe = _pipeline(*self.after_hooks)
        
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
