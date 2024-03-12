import functools
import logging
from collections import OrderedDict
from typing import Callable, Hashable, List, Dict, Any

import numpy as np
from tqdm.auto import tqdm

from brainio.assemblies import NeuroidAssembly, walk_coords
from brainscore_vision.model_helpers.utils import fullname

from brainscore_vision.model_helpers.activations.temporal.core.executor import BatchExecutor
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
    max_spatial_size: int
        the maximum spatial size of the activations. If the spatial size of the activations is larger than this value,
        the activations will be downsampled to this size. This is used to avoid the large memory consumption by the first layers of some model.
    batch_size: int
        number of stimuli to process in each batch.
    batch_grouper: function
        function that takes a stimulus and return the property based on which the stimuli can be grouped.
    batch_padding: bool
        whether to pad the each batch with the last stimulus to make it the same size as the specified batch size.
        Otherwise, some batches will have size < batch_size because of the lacking of samples in that group.
    dtype: np.dtype
        data type of the activations.

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
            stimulus_type : Stimulus,
            layer_activation_format : dict = None,
            max_spatial_size : int = None,
            batch_size : int = 64,
            batch_grouper : Callable[[Stimulus], Hashable] = None,
            batch_padding : bool = False,
            dtype : np.dtype = np.float16,
        ):

        self.stimulus_type = stimulus_type
        self.layer_activation_format = layer_activation_format
        self.max_spatial_size = max_spatial_size
        assert max_spatial_size is None, NotImplementedError("max_spatial_size is not implemented yet")
        self.dtype = dtype
        self._executor = BatchExecutor(get_activations, preprocessing, batch_size, batch_padding, batch_grouper, dtype)
        self._stimulus_set_hooks = {}
        self._batch_activations_hooks = {}
        self._logger = logging.getLogger(fullname(self))

    @property
    def identifier(self):
        to_add = []
        if self.max_spatial_size is not None:
            to_add.append(f".max_spatial={self.max_spatial_size}")
        to_add = "".join(to_add)
        return f"{self.__class__.__name__}{to_add}"

    def __call__(self, paths, layers):
        stimuli = self.convert_paths(paths)
        layer_activations = self.inference(stimuli, layers)
        layer_assemblies = OrderedDict()
        for layer in tqdm(layers, desc="Packaging layers"):
            layer_assemblies[layer] = self.package_layer(layer_activations[layer], layer, self.layer_activation_format[layer], stimuli)
        model_assembly = self.package(layer_assemblies, paths)
        breakpoint()
        return model_assembly

    # List[path] -> List[Stimulus] 
    def convert_paths(self, paths):
        return [self.stimulus_type.from_path(p) for p in paths]
    
    # List[Stimulus] -> Dict[layer: List[activation]]
    def inference(self, stimuli, layers):
        self._executor.add_stimuli(stimuli)
        return self._executor.execute(layers)
    
    # np.array -> NeuroidAssembly
    def package_layer(self, layer_activation, layer, layer_spec, stimuli):
        assert len(layer_activation) == len(stimuli)
        channels = self._map_dims(layer_spec)
        assembly = self._simple_package(layer_activation, ["stimulus_path"] + channels)
        assembly = self._stack_neuroid(assembly, channels)
        assembly = NeuroidAssembly(assembly)  # re-gather
        return assembly
    
    # Dict[layer: Assembly] -> NeuroidAssembly
    def package(self, layer_assemblies, stimuli_paths):
        # merge manually instead of using merge_data_arrays since `xarray.merge` is very slow with these large arrays
        # complication: (non)neuroid_coords are taken from the structure of layer_assemblies[0] i.e. the 1st assembly;
        # using these names/keys for all assemblies results in KeyError if the first layer contains dims
        # (see _package_layer) not present in later layers, e.g. first layer = conv, later layer = transformer layer
        self._logger.debug(f"Merging {len(layer_assemblies)} layer assemblies")
        layer_assemblies = {layer: self._add_neuroid_meta(assembly, layer) 
                            for layer, assembly in layer_assemblies.items()}
        layer_assemblies = list(layer_assemblies.values())
        layer_assemblies = [asm.transpose(*layer_assemblies[0].dims) for asm in layer_assemblies]
        model_assembly = np.concatenate([a.values for a in layer_assemblies],
                                        axis=layer_assemblies[0].dims.index('neuroid')).astype(self.dtype)
        
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

        # add stimulus_paths
        nonneuroid_coords["stimulus_path"] = ('stimulus_path', stimuli_paths)

        model_assembly = type(layer_assemblies[0])(model_assembly, coords={**nonneuroid_coords, **neuroid_coords},dims=layer_assemblies[0].dims)
        return model_assembly

    # add additional coords for the neuroid dim
    def _add_neuroid_meta(self, assembly, layer):
        num_neuroid = assembly.sizes["neuroid"]
        if "neuroid" in assembly.coords: 
            assembly = assembly.reset_index("neuroid")
        assembly = assembly.assign_coords({
                  'neuroid_num': ('neuroid', list(range(num_neuroid))),
                  'layer': ('neuroid', [layer] * num_neuroid),
                #   'model': ('neuroid', [self.identifier] * num_neuroid),
                })
        neuroid_id = [".".join([f"{value}" for value in values]) for values in zip(*[
            assembly[coord].values for coord in ['layer', 'neuroid_num']])]
        assembly = assembly.assign_coords(neuroid_id=('neuroid', neuroid_id))
        return NeuroidAssembly(assembly)

    # map dims to channel names
    @staticmethod
    def _map_dims(dims):
        return [channel_name_mapping[dim] for dim in dims]

    # package a numpy array to a NeuroidAssembly with given dims
    # make coord for each dimension to be the range of the dimension, as 0, 1, 2, ..., dim_size
    @staticmethod
    def _simple_package(activation: np.array, channel_names):
        dims = channel_names
        coords = {dim: range(activation.shape[i]) for i, dim in enumerate(dims)}
        return NeuroidAssembly(activation, coords=coords, dims=dims)
    
    # def _spatial_downsample(self, packaged_activation):
    #     if self.max_spatial_size is None:
    #         return packaged_activation

    # stack the channel dimensions to form the "neuroid" dimension
    @staticmethod
    def _stack_neuroid(assembly, channels):
        asm_cls = assembly.__class__
        assembly = assembly.stack(neuroid=channels).reset_index('neuroid')
        return asm_cls(assembly)


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
