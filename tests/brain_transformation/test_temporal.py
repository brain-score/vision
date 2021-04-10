import numpy as np

from brainio_base.assemblies import NeuroidAssembly
from model_tools.brain_transformation import TemporalIgnore


class LayerMappedModelMock:
    def __init__(self):
        # attributes that TemporalIgnore copies for outside use
        self.region_layer_map = None
        self.activations_model = None
        self.start_task = None

    def start_recording(self, *args, **kwargs):
        pass

    def look_at(self, *args, **kwargs):
        return NeuroidAssembly([[1, 2, 3], [1, 2, 3]], coords={'image_id': ('presentation', ['image1', 'image2']),
                                                               'object_name': (
                                                                   'presentation', ['number', 'number']),
                                                               'neuroid_id': ('neuroid', [1, 2, 3]),
                                                               'region': ('neuroid', ['IT'] * 3), },
                               dims=['presentation', 'neuroid'])


class TestTemporalIgnore:
    def test_single_timebin(self):
        model = TemporalIgnore(layer_model=LayerMappedModelMock())
        model.start_recording(recording_target='IT', time_bins=[(70, 170)])
        recordings = model.look_at('dummy')
        assert set(recordings.dims) == {'presentation', 'neuroid'}  # squeezed time-bin

    def test_two_timebins(self):
        layer_model = LayerMappedModelMock()
        model = TemporalIgnore(layer_model=layer_model)
        model.start_recording(recording_target='IT', time_bins=[(70, 170), (170, 270)])
        recordings = model.look_at('dummy')
        assert set(recordings.dims) == {'presentation', 'neuroid', 'time_bin'}
        np.testing.assert_array_equal(recordings['time_bin_start'].values, [70, 170])
        np.testing.assert_array_equal(recordings['time_bin_end'].values, [170, 270])

    def test_18_timebins(self):
        model = TemporalIgnore(layer_model=LayerMappedModelMock())
        time_bins = [(70 + i * 10, 80 + i * 10) for i in range(18)]
        model.start_recording(recording_target='IT', time_bins=time_bins)
        recordings = model.look_at('dummy')
        assert set(recordings.dims) == {'presentation', 'neuroid', 'time_bin'}
        np.testing.assert_array_equal(recordings['time_bin_start'].values, [start for start, end in time_bins])
        np.testing.assert_array_equal(recordings['time_bin_end'].values, [end for start, end in time_bins])
