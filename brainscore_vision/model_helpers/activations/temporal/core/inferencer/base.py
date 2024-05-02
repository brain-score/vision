import functools
import logging
from collections import OrderedDict
from typing import Callable, Hashable, List, Dict, Any, Union
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import gc

from brainio.assemblies import NeuroidAssembly, walk_coords
from brainscore_vision.model_helpers.utils import fullname

from brainscore_vision.model_helpers.activations.temporal.core.executor import BatchExecutor
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding, batch_2d_resize
from brainscore_vision.model_helpers.activations.temporal.inputs import Stimulus


channel_name_mapping = {
    "T": "channel_temporal",
    "C": "channel",
    "H": "channel_y",
    "W": "channel_x",
    "K": "channel_token"
}


class Inferencer:
    """Inferencer for batch processing of stimuli and packaging the activations.
    
    Parameters
    ----------
    get_activations : function 
        function that takes a list of processed stimuli and a list of layers, and returns a dictionary of activations.
    preprocessing : function
        function that takes a stimulus and returns a processed stimulus.
    stimulus_type: Stimulus
        the type of the stimulus.
    layer_activation_format: dict
        a dictionary that specifies the dimensions of the activations of each layer.
        For example, {"temp_conv": "TCHW", "spatial_conv": "CHW",  "fc": "C"}.
    visual_degrees: float
        the visual degrees of the stimuli.
    max_spatial_size: int/float
        the maximum spatial size of the activations. If the spatial size of the activations is larger than this value,
        the activations will be downsampled to this size. This is used to avoid the large memory consumption by the first layers of some model.
        If float, resize the image based on this factor.
    dtype: np.dtype
        data type of the activations.
    batch_size: int
        number of stimuli to process in each batch.
    batch_grouper: function
        function that takes a stimulus and return the property based on which the stimuli can be grouped.
    batch_padding: bool
        whether to pad the each batch with the last stimulus to make it the same size as the specified batch size.
        Otherwise, some batches will have size < batch_size because of the lacking of samples in that group.
    max_workers: int
        the maximum number of workers to use for parallel processing.

        
    APIs
    ----
    __call__(paths, layers)
        process the stimuli and return a holistic assembly that compile activations of all specified layers.
        The returned assembly is a NeuroidAssembly with the dimensions [stimulus_path, neuroid].
        All dimensions of the activations will be stacked together to form the "neuroid" dimension.

    Examples
    --------
    >>> inferencer = Inferencer(get_activations, preprocessing, Video, batch_size=64, batch_grouper=lambda s: s.duration)
    >>> model_assembly = inferencer(stimulus_paths, layers)
    """

    def __init__(
            self, 
            get_activations : Callable[[List[Any]], Dict[str, np.array]], 
            preprocessing : Callable[[List[Stimulus]], Any],
            layer_activation_format : dict,
            stimulus_type : Stimulus,
            visual_degrees : float = 8.,
            max_spatial_size : Union[int, float] = None,
            dtype : np.dtype = np.float16,
            batch_size : int = 64,
            batch_grouper : Callable[[Stimulus], Hashable] = None,
            batch_padding : bool = False,
            max_workers : int = None,
            *args,
            **kwargs
        ):

        self.stimulus_type = stimulus_type
        self.layer_activation_format = layer_activation_format
        if isinstance(max_spatial_size, float):
            assert max_spatial_size < 1, "a proporational max_spatial_size should be < 1."
        self.max_spatial_size = max_spatial_size
        self.visual_degrees = visual_degrees
        self.dtype = dtype
        self._executor = BatchExecutor(get_activations, preprocessing, batch_size, batch_padding, batch_grouper, max_workers)
        self._stimulus_set_hooks = {}
        self._batch_activations_hooks = {}
        self._logger = logging.getLogger(fullname(self))

        # register hooks
        self._executor.register_after_hook(self._make_spatial_downsample_hook(max_spatial_size))
        self._executor.register_after_hook(self._make_dtype_hook(dtype))

    @property
    # identifier for the inferencer: including all the features that may affect the activations
    def identifier(self) -> str:
        to_add = [
            f".dtype={self.dtype.__name__}",
            f".vdeg={self.visual_degrees}",
        ]
        if self.max_spatial_size is not None:
            to_add.append(f".max_s={self.max_spatial_size}")
        to_add = "".join(to_add)
        return f"{self.__class__.__name__}{to_add}"
    
    def set_visual_degrees(self, visual_degrees: float):
        self.visual_degrees = visual_degrees

    # given the paths of the stimuli and the layers, return the model activations as a NeuroidAssembly
    def __call__(self, paths: List[Union[str, Path]], layers: List[str]):
        stimuli = self.load_stimuli(paths)
        layer_activations = self.inference(stimuli, layers)
        layer_assemblies = OrderedDict()
        for layer in tqdm(layers, desc="Packaging layers"):
            layer_assemblies[layer] = self.package_layer(layer_activations[layer], self.layer_activation_format[layer], stimuli)
            del layer_activations[layer]
            gc.collect()  # reduce memory usage
        model_assembly = self.package(layer_assemblies, paths)
        return model_assembly

    def load_stimuli(self, paths : List[Union[str, Path]]) -> List[Stimulus]:
        ret = []
        for p in tqdm(paths, desc="Loading stimuli"):
            ret.append(self.load_stimulus(p))
        return ret

    def load_stimulus(self, path : Union[str, Path]) -> Stimulus:
        return self.stimulus_type.from_path(path)
    
    # process the list of stimulus and return the activations (list of np.array, 
    # whose length is the number of stimuli) of the specified layers
    def inference(self, stimuli : List[Stimulus], layers : List[str]) -> Dict[str, List[np.array]]:
        self._executor.add_stimuli(stimuli)
        return self._executor.execute(layers)
    
    # Take the layer_activation (a list of np.array) and the layer specification,
    # and package them into a NeuroidAssembly with all channels flattened into the "neuroid" dimension
    def package_layer(
            self, 
            layer_activation : List[np.array],
            layer_spec : str, 
            stimuli : List[Stimulus]
        ):
        assert len(layer_activation) == len(stimuli)
        layer_activation = stack_with_nan_padding(layer_activation, dtype=self.dtype)
        channels = self._map_dims(layer_spec)
        assembly = self._package(layer_activation, ["stimulus_path"] + channels)
        assembly = self._stack_neuroid(assembly, channels)
        return assembly
    
    # package the assemblies from different layers into a single one by concat along the neuroid dimension
    def package(self, layer_assemblies : Dict[str, NeuroidAssembly], stimuli_paths : List[str]) -> NeuroidAssembly:
        # merge manually instead of using merge_data_arrays since `xarray.merge` is very slow with these large arrays
        # complication: (non)neuroid_coords are taken from the structure of layer_assemblies[0] i.e. the 1st assembly;
        # using these names/keys for all assemblies results in KeyError if the first layer contains dims
        # (see _package_layer) not present in later layers, e.g. first layer = conv, later layer = transformer layer
        self._logger.debug(f"Merging {len(layer_assemblies)} layer assemblies")
        layers = list(layer_assemblies.keys())
        layer_assemblies = list(layer_assemblies.values())
        layer_assemblies = [asm.transpose(*layer_assemblies[0].dims) for asm in layer_assemblies]
        
        nonneuroid_coords = {coord: (dims, values) for coord, dims, values in walk_coords(layer_assemblies[0])
                             if set(dims) != {'neuroid'}}
        neuroid_coords = [(coord, dims) for layer_assembly in layer_assemblies for coord, dims, values in walk_coords(layer_assembly)
                             if set(dims) == {'neuroid'} and coord!='neuroid']
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

        # add stimulus_paths
        nonneuroid_coords["stimulus_path"] = ('stimulus_path', stimuli_paths)

        # add layer, neuroid_num, neuroid_id
        layer_sizes = [asm.sizes['neuroid'] for asm in layer_assemblies]
        neuroid_coords['layer'] = (('neuroid',), np.concatenate([[layer] * size for layer, size in zip(layers, layer_sizes)]))
        neuroid_coords['neuroid_num'] = (('neuroid',), np.concatenate([np.arange(size) for size in layer_sizes]))
        neuroid_coords['neuroid_id'] = (('neuroid',), np.concatenate([np.array([f"{layer}.{neuroid_num}" for neuroid_num in range(size)]) 
                                                                   for layer, size in zip(layers, layer_sizes)]))

        model_assembly = np.concatenate([a.values for a in layer_assemblies], axis=layer_assemblies[0].dims.index('neuroid'))
        model_assembly = type(layer_assemblies[0])(model_assembly, coords={**nonneuroid_coords, **neuroid_coords},dims=layer_assemblies[0].dims)
        return model_assembly
    
    # turn the activations into the specified dtype
    def _make_dtype_hook(self, dtype):
        return lambda val, layer, stimulus: val.astype(dtype)
    
    # downsample the activations with the largest spatial size (among width and height) to the specified size
    def _make_spatial_downsample_hook(self, max_spatial_size, mode="pool"):
        def hook(val, layer, stimulus):
            if max_spatial_size is None:
                return val
            
            dims = self.layer_activation_format[layer]

            # require both H and W dimensions to do spatial downsampling
            if "H" not in dims or "W" not in dims:
                return val

            H_dim, W_dim = dims.index("H"), dims.index("W")
            val = val.swapaxes(H_dim, 0).swapaxes(W_dim, 1)
            shape = val.shape[2:]
            h, w = val.shape[:2]
            val = val.reshape(h, w, -1)
            new_size = _compute_new_size(w, h, self.max_spatial_size)
            new_val = batch_2d_resize(val[None,:], new_size, mode=mode)[0]
            new_val = new_val.reshape(*new_size, *shape)
            new_val = new_val.swapaxes(0, H_dim).swapaxes(1, W_dim)
            return new_val.astype(self.dtype)
        return hook

    # map dims to channel names
    @staticmethod
    def _map_dims(dims):
        return [channel_name_mapping[dim] for dim in dims]

    # stack the channel dimensions to form the "neuroid" dimension
    @staticmethod
    def _stack_neuroid(assembly, channels):
        asm_cls = assembly.__class__
        assembly = assembly.stack(neuroid=channels)
        return asm_cls(assembly)

    # package an activation numpy array into a NeuroidAssembly with specified dims
    @staticmethod
    def _package(activation: np.array, dims):
        coords = {dim: range(activation.shape[i]) for i, dim in enumerate(dims)}
        ret = NeuroidAssembly(activation, coords=coords, dims=dims)
        return ret

def _compute_new_size(w, h, max_spatial_size):
    if isinstance(max_spatial_size, int):
        if h > w:
            new_h = max_spatial_size
            new_w = int(w * new_h / h)
        else:
            new_w = max_spatial_size
            new_h = int(h * new_w / w)
    else:
        new_h = int(h * max_spatial_size)
        new_w = int(w * max_spatial_size)
    
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    return new_h, new_w
