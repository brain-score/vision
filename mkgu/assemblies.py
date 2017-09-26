from __future__ import absolute_import, division, print_function, unicode_literals

from xarray import DataArray, Dataset

from mkgu import fetch


class DataPoint(object):
    """A DataPoint represents one value, usually a recording from one neuron or node,
    in response to one presentation of a stimulus.  """
    def __init__(self, value, neuroid, presentation):
        self.value = value
        self.neuroid = neuroid
        self.presentation = presentation


class DataAssembly(object):
    """A DataAssembly represents a set of data a researcher wishes to work with for
    an analysis or benchmarking task.  """
    def __init__(self, name="HvM", xr_data=None):
        self.name = name
        self.xr_data = xr_data

    def __getattr__(self, attr):
        return getattr(self.xr_data, attr)

    def __setattr__(self, attr, val):
        if attr == 'xr_data':
            object.__setattr__(self, attr, val)
        return setattr(self.xr_data, attr, val)

    def coords_for_dim(self, dim):
        return [x[0] for x in self.coords.variables.items() if x[1].dims == (dim,)]

    def gather_indexes(self):
        return self.set_index(append=True, **{dim: self.coords_for_dim(self, dim) for dim in self.dims})

    @classmethod
    def get_assembly(cls, name):
        assy = fetch.get_lookup().lookup_assembly(name)


class BehavioralAssembly(DataAssembly):
    """A BehavioralAssembly is a DataAssembly containing behavioral data.  """
    def __init__(self, **kwargs):
        super(BehavioralAssembly, self).__init__(**kwargs)


class NeuroidAssembly(DataAssembly):
    """A NeuroidAssembly is a DataAssembly containing data recorded from either neurons
    or neuron analogues.  """
    def __init__(self, **kwargs):
        super(NeuroidAssembly, self).__init__(**kwargs)


class NeuronRecordingAssembly(NeuroidAssembly):
    """A NeuronRecordingAssembly is a NeuroidAssembly containing data recorded from neurons.  """
    def __init__(self, **kwargs):
        super(NeuronRecordingAssembly, self).__init__(**kwargs)


class ModelFeaturesAssembly(NeuroidAssembly):
    """A ModelFeaturesAssembly is a NeuroidAssembly containing data captured from nodes in
    a machine learning model.  """
    def __init__(self, **kwargs):
        super(ModelFeaturesAssembly, self).__init__(**kwargs)





