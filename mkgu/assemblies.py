from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import operator

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset
from xarray.core.groupby import GroupBy

from mkgu import fetch


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
        if not isinstance(group_coord_names, (tuple, list)):
            group_coord_names = [group_coord_names]
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


class BehavioralAssembly(DataAssembly):
    """A BehavioralAssembly is a DataAssembly containing behavioral data.  """
    def __init__(self, *args, **kwargs):
        super(BehavioralAssembly, self).__init__(*args, **kwargs)


class NeuroidAssembly(DataAssembly):
    """A NeuroidAssembly is a DataAssembly containing data recorded from either neurons
    or neuron analogues.  """
    def __init__(self, *args, **kwargs):
        super(NeuroidAssembly, self).__init__(*args, **kwargs)


class NeuronRecordingAssembly(NeuroidAssembly):
    """A NeuronRecordingAssembly is a NeuroidAssembly containing data recorded from neurons.  """
    def __init__(self, *args, **kwargs):
        super(NeuronRecordingAssembly, self).__init__(*args, **kwargs)


class ModelFeaturesAssembly(NeuroidAssembly):
    """A ModelFeaturesAssembly is a NeuroidAssembly containing data captured from nodes in
    a machine learning model.  """
    def __init__(self, *args, **kwargs):
        super(ModelFeaturesAssembly, self).__init__(*args, **kwargs)


def coords_for_dim(xr_data, dim, exclude_indexes=True):
    result = []
    for x in xr_data.coords.variables.items():
        only_this_dim = x[1].dims == (dim,)
        exclude_because_index = exclude_indexes and isinstance(x[1], xr.IndexVariable)
        if only_this_dim and not exclude_because_index:
            result.append(x[0])
    return result


def gather_indexes(xr_data):
    """This is only necessary as long as xarray cannot persist MultiIndex to netCDF.  """
    coords_d = {}
    for dim in xr_data.dims:
        coords = coords_for_dim(xr_data, dim)
        if coords:
            coords_d[dim] = coords
    if coords_d:
        xr_data.set_index(append=True, inplace=True, **coords_d)
    return xr_data


class GroupbyBridge(object):
    """Wraps an xarray GroupBy object to allow grouping on multiple coordinates.   """
    def __init__(self, groupby, assembly, dim, group_coord_names, delimiter, multi_group_name):
        self.groupby = groupby
        self.assembly = assembly
        self.dim = dim
        self.group_coord_names =  group_coord_names
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
        split_coords = np.array(list(map(lambda s: s.split(self.delimiter), result.coords[self.multi_group_name].values))).T
        for coord_name, coord in zip(self.group_coord_names, split_coords):
            result.coords[coord_name] = (self.multi_group_name, coord)
        result.reset_index(self.multi_group_name, drop=True, inplace=True)
        result.set_index(append=True, inplace=True, **{self.multi_group_name: self.group_coord_names})
        result = result.rename({self.multi_group_name: self.dim})
        return result


class GroupbyError(Exception):
    pass


