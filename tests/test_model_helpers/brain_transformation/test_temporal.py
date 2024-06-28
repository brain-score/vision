import numpy as np
import pytest

from brainio.assemblies import NeuroidAssembly
from brainscore_vision.model_helpers.brain_transformation import TemporalAligned
from brainscore_vision.model_interface import BrainModel


class LayerMappedModelMock:
    def __init__(self, output_temporal=False):
        # attributes that TemporalAligned copies for outside use
        self.region_layer_map = None
        self.activations_model = None
        self.start_task = None
        self.output_temporal = output_temporal

    def start_recording(self, *args, **kwargs):
        pass

    def look_at(self, *args, **kwargs):
        if self.output_temporal:
            return NeuroidAssembly([[[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]]], 
                                coords={'stimulus_id': ('presentation', ['image1', 'image2']),
                                                        'object_name': (
                                                            'presentation', ['number', 'number']),
                                                        'neuroid_id': ('neuroid', [1, 2, 3]),
                                                        'region': ('neuroid', ['IT'] * 3), 
                                                        'time_bin_start': ('time_bin', [0, 150]), 
                                                        'time_bin_end': ('time_bin', [150, 300]), },
                                dims=['presentation', 'neuroid', 'time_bin'])
        else:
            return NeuroidAssembly([[1, 2, 3], [1, 2, 3]], coords={'stimulus_id': ('presentation', ['image1', 'image2']),
                                                                'object_name': (
                                                                    'presentation', ['number', 'number']),
                                                                'neuroid_id': ('neuroid', [1, 2, 3]),
                                                                'region': ('neuroid', ['IT'] * 3), },
                                dims=['presentation', 'neuroid'])


class TestTemporalAligned:
    @pytest.mark.parametrize('output_temporal', [False, True])
    def test_single_timebin(self, output_temporal):
        model = TemporalAligned(layer_model=LayerMappedModelMock(output_temporal))
        model.start_recording(recording_target=BrainModel.RecordingTarget.IT, time_bins=[(70, 170)])
        recordings = model.look_at('dummy')
        assert set(recordings.dims) == {'presentation', 'neuroid'}  # squeezed time-bin

    @pytest.mark.parametrize('output_temporal', [False, True])
    def test_two_timebins(self, output_temporal):
        layer_model = LayerMappedModelMock(output_temporal)
        model = TemporalAligned(layer_model=layer_model)
        model.start_recording(recording_target=BrainModel.RecordingTarget.IT, time_bins=[(70, 170), (170, 270)])
        recordings = model.look_at('dummy')
        assert set(recordings.dims) == {'presentation', 'neuroid', 'time_bin'}
        np.testing.assert_array_equal(recordings['time_bin_start'].values, [70, 170])
        np.testing.assert_array_equal(recordings['time_bin_end'].values, [170, 270])

    @pytest.mark.parametrize('output_temporal', [False, True])
    def test_18_timebins(self, output_temporal):
        model = TemporalAligned(layer_model=LayerMappedModelMock(output_temporal))
        time_bins = [(70 + i * 10, 80 + i * 10) for i in range(18)]
        model.start_recording(recording_target=BrainModel.RecordingTarget.IT, time_bins=time_bins)
        recordings = model.look_at('dummy')
        assert set(recordings.dims) == {'presentation', 'neuroid', 'time_bin'}
        np.testing.assert_array_equal(recordings['time_bin_start'].values, [start for start, end in time_bins])
        np.testing.assert_array_equal(recordings['time_bin_end'].values, [end for start, end in time_bins])
