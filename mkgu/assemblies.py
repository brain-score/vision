from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import operator
from collections import OrderedDict, defaultdict

import numpy as np
import peewee
import xarray as xr
from xarray import DataArray

from mkgu.lookup import pwdb
from mkgu.stimuli import StimulusSetModel


class DataPoint(object):
    """A DataPoint represents one value, usually a recording from one neuron or node,
    in response to one presentation of a stimulus.  """

    def __init__(self, value, neuroid, presentation):
        self.value = value
        self.neuroid = neuroid
        self.presentation = presentation


class DataAssembly(DataArray):
    """A DataAssembly represents a set of data a researcher wishes to work with for
    an analysis or benchmarking task.  """

    def __init__(self, *args, **kwargs):
        super(DataAssembly, self).__init__(*args, **kwargs)
        gather_indexes(self)

    def multi_groupby(self, group_coord_names, *args, **kwargs):
        delimiter = "|"
        multi_group_name = "multi_group"
        dim = self._dim_of_group_coords(group_coord_names)
        tmp_assy = self._join_group_coords(dim, group_coord_names, delimiter, multi_group_name)
        result = tmp_assy.groupby(multi_group_name, *args, **kwargs)
        return GroupbyBridge(result, self, dim, group_coord_names, delimiter, multi_group_name)

    def _join_group_coords(self, dim, group_coord_names, delimiter, multi_group_name):
        tmp_assy = self.copy()
        group_coords = [tmp_assy.coords[c] for c in group_coord_names]
        to_join = [x for y in group_coords for x in (y, delimiter)][:-1]
        tmp_assy.coords[multi_group_name] = functools.reduce(operator.add, to_join)
        tmp_assy.set_index(append=True, inplace=True, **{dim: multi_group_name})
        return tmp_assy

    def _dim_of_group_coords(self, group_coord_names):
        dimses = [self.coords[coord_name].dims for coord_name in group_coord_names]
        dims = [dim for dim_tuple in dimses for dim in dim_tuple]
        if len(set(dims)) == 1:
            return dims[0]
        else:
            raise GroupbyError("All coordinates for grouping must be associated with the same single dimension.  ")

    def multisel(self, method=None, tolerance=None, drop=False, **indexers):
        """
        partial workaround to keep multi-indexes and scalar coords
        https://github.com/pydata/xarray/issues/1491, https://github.com/pydata/xarray/pull/1426

        this method might slow things down, use with caution
        """
        indexer_dims = {index: self[index].dims for index in indexers}
        dims = []
        for _dims in indexer_dims.values():
            assert len(_dims) == 1
            dims.append(_dims[0])
        coords_dim, dim_coords = {}, defaultdict(list)
        for dim in dims:
            for coord, coord_dims, _ in walk_coords(self):
                if array_is_element(coord_dims, dim):
                    coords_dim[coord] = dim
                    dim_coords[dim].append(coord)

        result = super().sel(method, tolerance, drop, **indexers)

        # stack back together
        for result_dim in result.dims:
            if result_dim not in self.dims:
                original_dim = coords_dim[result_dim]
                result = result.stack(**{original_dim: [coord for coord in dim_coords[original_dim]
                                                        if hasattr(result, coord)]})
        # add scalar indexer variable
        for index, value in indexers.items():
            dim = indexer_dims[index]
            assert len(dim) == 1
            value = np.repeat(value, len(result[dim[0]]))
            result[index] = dim, value
        return result


class BehavioralAssembly(DataAssembly):
    """A BehavioralAssembly is a DataAssembly containing behavioral data.  """
    pass


class NeuroidAssembly(DataAssembly):
    """A NeuroidAssembly is a DataAssembly containing data recorded from either neurons
    or neuron analogues.  """
    pass


class NeuronRecordingAssembly(NeuroidAssembly):
    """A NeuronRecordingAssembly is a NeuroidAssembly containing data recorded from neurons.  """
    pass


class ModelFeaturesAssembly(NeuroidAssembly):
    """A ModelFeaturesAssembly is a NeuroidAssembly containing data captured from nodes in
    a machine learning model.  """
    pass


def coords_for_dim(xr_data, dim, exclude_indexes=True):
    result = OrderedDict()
    for key, value in xr_data.coords.variables.items():
        only_this_dim = value.dims == (dim,)
        exclude_because_index = exclude_indexes and isinstance(value, xr.IndexVariable)
        if only_this_dim and not exclude_because_index:
            result[key] = value
    return result


def gather_indexes(xr_data):
    """This is only necessary as long as xarray cannot persist MultiIndex to netCDF.  """
    coords_d = {}
    for dim in xr_data.dims:
        coords = coords_for_dim(xr_data, dim)
        if coords:
            coords_d[dim] = list(coords.keys())
    if coords_d:
        xr_data.set_index(append=True, inplace=True, **coords_d)
    return xr_data


class GroupbyBridge(object):
    """Wraps an xarray GroupBy object to allow grouping on multiple coordinates.   """

    def __init__(self, groupby, assembly, dim, group_coord_names, delimiter, multi_group_name):
        self.groupby = groupby
        self.assembly = assembly
        self.dim = dim
        self.group_coord_names = group_coord_names
        self.delimiter = delimiter
        self.multi_group_name = multi_group_name

    def __getattr__(self, attr):
        result = getattr(self.groupby, attr)
        if callable(result):
            result = self.wrap_groupby(result)
        return result

    def wrap_groupby(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, type(self.assembly)):
                result = self.split_group_coords(result)
            return result

        return wrapper

    def split_group_coords(self, result):
        split_coords = np.array(
            list(map(lambda s: s.split(self.delimiter), result.coords[self.multi_group_name].values))).T
        for coord_name, coord in zip(self.group_coord_names, split_coords):
            result.coords[coord_name] = (self.multi_group_name, coord)
        result.reset_index(self.multi_group_name, drop=True, inplace=True)
        result.set_index(append=True, inplace=True, **{self.multi_group_name: self.group_coord_names})
        result = result.rename({self.multi_group_name: self.dim})
        return result


class GroupbyError(Exception):
    pass


class AssemblyModel(peewee.Model):
    """An AssemblyModel stores information about the canonical location where the data
    for a DataAssembly is stored.  """
    name = peewee.CharField()
    assembly_class = peewee.CharField()
    stimulus_set = peewee.ForeignKeyField(StimulusSetModel, backref="assembly_models")

    class Meta:
        database = pwdb


class AssemblyStoreModel(peewee.Model):
    """An AssemblyStoreModel stores the location of a DataAssembly data file.  """
    assembly_type = peewee.CharField()
    location_type = peewee.CharField()
    location = peewee.CharField()

    class Meta:
        database = pwdb


class AssemblyStoreMap(peewee.Model):
    """An AssemblyStoreMap links an AssemblyRecord to an AssemblyStore.  """
    assembly_model = peewee.ForeignKeyField(AssemblyModel, backref="assembly_store_maps")
    assembly_store_model = peewee.ForeignKeyField(AssemblyStoreModel, backref="assembly_store_maps")
    role = peewee.CharField()

    class Meta:
        database = pwdb


class AssemblyLookupError(Exception):
    pass


def lookup_assembly(name):
    pwdb.connect(reuse_if_open=True)
    try:
        assy = AssemblyModel.get(AssemblyModel.name == name)
    except AssemblyModel.DoesNotExist as e:
        raise AssemblyLookupError("A DataAssembly named " + name + " was not found.")
    return assy


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
        if is_index:
            for level in values.variable.level_names:
                level_values = assembly.coords[level]
                yield level, level_values.dims, level_values.values
        else:
            yield name, values.dims, values.values
    return coords
