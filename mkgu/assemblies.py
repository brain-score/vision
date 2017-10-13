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


class DataAssembly(DataArray):
    """A DataAssembly represents a set of data a researcher wishes to work with for
    an analysis or benchmarking task.  """

    def __init__(self, *args, **kwargs):
        super(DataAssembly, self).__init__(*args, **kwargs)


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





