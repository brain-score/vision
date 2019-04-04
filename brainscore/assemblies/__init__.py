from collections import OrderedDict

import numpy as np
import xarray as xr
from numpy.random.mtrand import RandomState
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from xarray import DataArray


class AssemblyLoader:
    def __init__(self, name):
        self.name = name

    def __call__(self):
        raise NotImplementedError()


def split_assembly(assembly, on='image_id', stratification='object_name',
                   named_ratios=OrderedDict([('map', .8), ('test', .2)]), rng=None):
    from brainscore.metrics.transformations import subset  # avoid circular import
    dim = assembly[on].dims[0]
    rng = rng or RandomState(seed=1)
    _, unique_indices = np.unique(assembly[on].values, return_index=True)
    unique_indices = np.sort(unique_indices)  # preserve order
    choice_options = assembly[on].values[unique_indices]
    stratification_values = assembly[stratification].values[unique_indices] \
        if hasattr(assembly, stratification) else None

    assemblies = []
    for name, ratio in named_ratios.items():
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=ratio, test_size=None, random_state=rng) \
            if stratification_values is not None else \
            ShuffleSplit(n_splits=1, train_size=ratio, test_size=None, random_state=rng)
        split_args = [np.zeros(len(choice_options))] + \
                     ([stratification_values] if stratification_values is not None else [])
        split = list(splitter.split(*split_args))
        assert len(split) == 1
        choice_indices, leftover_indices = split[0]
        choice = choice_options[choice_indices]
        # reduce choices for next iteration
        choice_options = choice_options[leftover_indices]
        stratification_values = stratification_values[leftover_indices] if stratification_values is not None else None
        # build assembly
        subset_indexer = DataArray(np.zeros(len(choice)), coords={on: choice}, dims=[on]).stack(**{dim: [on]})
        choice_assembly = subset(assembly, subset_indexer, dims_must_match=False)
        choice_assembly.attrs['stimulus_set'] = assembly.stimulus_set[
            assembly.stimulus_set['image_id'].isin(choice_assembly['image_id'].values)]
        choice_assembly.stimulus_set.name = assembly.stimulus_set.name + "_" + name
        assemblies.append(choice_assembly)
    return assemblies


def merge_data_arrays(data_arrays):
    # https://stackoverflow.com/a/50125997/2225200
    merged = xr.merge([similarity.rename('z') for similarity in data_arrays])['z'].rename(None)
    # ensure same class
    return type(data_arrays[0])(merged)


def array_is_element(arr, element):
    return len(arr) == 1 and arr[0] == element


def walk_coords(assembly):
    """
    walks through coords and all levels, just like the `__repr__` function, yielding `(name, dims, values)`.
    """
    coords = {}

    for name, values in assembly.coords.items():
        # partly borrowed from xarray.core.formatting#summarize_coord
        is_index = name in assembly.dims
        if is_index and values.variable.level_names:
            for level in values.variable.level_names:
                level_values = assembly.coords[level]
                yield level, level_values.dims, level_values.values
        else:
            yield name, values.dims, values.values
    return coords
