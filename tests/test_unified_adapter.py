import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import MagicMock, PropertyMock, patch

from brainscore_core.model_interface import TaskContext, UnifiedModel, BrainScoreModel
from brainscore_core.streaming_helpers import score_stimuli
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision.compat.unified_adapter import VisionModelAdapter


def _make_legacy_model(identifier='mock-resnet', visual_degrees=8,
                       region_layer_map=None):
    """Create a mock legacy BrainModel."""
    legacy = MagicMock()
    # identifier is a @property in vision
    type(legacy).identifier = PropertyMock(return_value=identifier)
    legacy.visual_degrees.return_value = visual_degrees
    legacy.look_at.return_value = 'mock_assembly'
    legacy.start_task.return_value = None
    legacy.start_recording.return_value = None

    if region_layer_map is not None:
        legacy.layer_model.region_layer_map = region_layer_map
    else:
        # No layer_model attribute
        del legacy.layer_model

    return legacy


def _vision_stimulus_set():
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': ['s0', 's1'],
        'image_path': ['s0.png', 's1.png'],
        'object_name': ['cat', 'dog'],
    }))
    stimuli.identifier = 'synthetic-vision'
    return stimuli


def _vision_neural_assembly():
    return NeuroidAssembly(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        coords={
            'stimulus_id': ('presentation', ['s0', 's1']),
            'object_name': ('presentation', ['cat', 'dog']),
            'neuroid_id': ('neuroid', ['layer4.0', 'layer4.1']),
            'layer': ('neuroid', ['layer4', 'layer4']),
        },
        dims=['presentation', 'neuroid'],
    )


class TestVisionAdapterIsUnifiedModel:

    def test_isinstance(self):
        adapter = VisionModelAdapter(_make_legacy_model())
        assert isinstance(adapter, UnifiedModel)


class TestVisionAdapterIdentity:

    def test_identifier_from_property(self):
        legacy = _make_legacy_model(identifier='cornet-s')
        adapter = VisionModelAdapter(legacy)
        assert adapter.identifier == 'cornet-s'

    def test_supported_modalities(self):
        adapter = VisionModelAdapter(_make_legacy_model())
        assert adapter.supported_modalities == {'vision'}

    def test_region_layer_map_from_layer_model(self):
        legacy = _make_legacy_model(region_layer_map={'V1': 'layer1', 'IT': 'layer4'})
        adapter = VisionModelAdapter(legacy)
        assert adapter.region_layer_map == {'V1': 'layer1', 'IT': 'layer4'}

    def test_region_layer_map_empty_when_no_layer_model(self):
        legacy = _make_legacy_model()
        adapter = VisionModelAdapter(legacy)
        assert adapter.region_layer_map == {}

    def test_visual_degrees(self):
        legacy = _make_legacy_model(visual_degrees=10)
        adapter = VisionModelAdapter(legacy)
        assert adapter.visual_degrees() == 10


class TestVisionAdapterProcess:

    def test_process_delegates_to_look_at(self):
        legacy = _make_legacy_model()
        legacy.look_at.return_value = 'neural_assembly'
        adapter = VisionModelAdapter(legacy)
        stimuli = MagicMock()

        result = adapter.process(stimuli)

        assert result == 'neural_assembly'
        legacy.look_at.assert_called_once_with(stimuli, 1)

    def test_process_passes_number_of_trials(self):
        legacy = _make_legacy_model()
        adapter = VisionModelAdapter(legacy)
        stimuli = MagicMock()

        adapter.process(stimuli, number_of_trials=10)

        legacy.look_at.assert_called_once_with(stimuli, 10)

    def test_process_returns_identical_output_to_look_at(self):
        """Core requirement: process() output == look_at() output."""
        legacy = _make_legacy_model()
        sentinel = object()
        legacy.look_at.return_value = sentinel
        adapter = VisionModelAdapter(legacy)
        stimuli = MagicMock()

        result = adapter.process(stimuli)
        assert result is sentinel

    def test_score_stimuli_interact_matches_legacy_look_at_exactly(self):
        stimuli = _vision_stimulus_set()
        expected = _vision_neural_assembly()

        legacy_expected = _make_legacy_model(region_layer_map={'IT': 'layer4'})
        legacy_expected.look_at.return_value = expected
        expected_adapter = VisionModelAdapter(legacy_expected)
        expected_adapter.start_recording('IT')
        legacy_output = expected_adapter.process(stimuli)

        legacy_stream = _make_legacy_model(region_layer_map={'IT': 'layer4'})
        legacy_stream.look_at.return_value = expected
        stream_adapter = VisionModelAdapter(legacy_stream)
        scored = score_stimuli(stream_adapter, stimuli, record='IT')

        xr.testing.assert_identical(scored, legacy_output)
        legacy_stream.start_recording.assert_called_once_with(
            'IT', [(70, 170)]
        )

class TestVisionAdapterLegacyMethods:

    def test_look_at_delegates(self):
        """Existing benchmarks call look_at() directly on the adapter."""
        legacy = _make_legacy_model()
        legacy.look_at.return_value = 'assembly'
        adapter = VisionModelAdapter(legacy)
        stimuli = MagicMock()

        result = adapter.look_at(stimuli, number_of_trials=5)

        assert result == 'assembly'
        legacy.look_at.assert_called_once_with(stimuli, 5)


class TestVisionAdapterStartTask:

    def test_start_task_unwraps_context(self):
        """New API: start_task(TaskContext(...))."""
        legacy = _make_legacy_model()
        adapter = VisionModelAdapter(legacy)
        fitting = MagicMock()
        ctx = TaskContext(
            task_type='probabilities',
            fitting_stimuli=fitting,
            label_set=['cat', 'dog'],
        )

        adapter.start_task(ctx)

        legacy.start_task.assert_called_once_with('probabilities', fitting)

    def test_start_task_passes_none_fitting_stimuli(self):
        legacy = _make_legacy_model()
        adapter = VisionModelAdapter(legacy)
        ctx = TaskContext(task_type='label')

        adapter.start_task(ctx)

        legacy.start_task.assert_called_once_with('label', None)

    def test_start_task_legacy_two_arg_call(self):
        """Existing benchmarks call start_task(task, fitting_stimuli) directly."""
        legacy = _make_legacy_model()
        adapter = VisionModelAdapter(legacy)
        fitting = MagicMock()

        adapter.start_task('probabilities', fitting)

        legacy.start_task.assert_called_once_with('probabilities', fitting)

    def test_start_task_legacy_with_kwargs(self):
        """Some benchmarks pass number_of_trials as kwarg."""
        legacy = _make_legacy_model()
        adapter = VisionModelAdapter(legacy)

        adapter.start_task('probabilities', 'fitting_data', number_of_trials=1)

        legacy.start_task.assert_called_once_with(
            'probabilities', 'fitting_data', number_of_trials=1
        )


class TestVisionAdapterStartRecording:

    def test_start_recording_with_explicit_time_bins(self):
        legacy = _make_legacy_model()
        adapter = VisionModelAdapter(legacy)

        adapter.start_recording('IT', time_bins=[(50, 100), (100, 150)])

        legacy.start_recording.assert_called_once_with('IT', [(50, 100), (100, 150)])

    def test_start_recording_default_time_bins(self):
        """Default time_bins for vision is [(70, 170)]."""
        legacy = _make_legacy_model()
        adapter = VisionModelAdapter(legacy)

        adapter.start_recording('V1')

        legacy.start_recording.assert_called_once_with('V1', [(70, 170)])


class TestVisionAutoWrapping:

    def test_load_model_wraps_legacy(self):
        """load_model() should wrap a legacy BrainModel in VisionModelAdapter."""
        import brainscore_vision
        legacy = _make_legacy_model(identifier='test-legacy')

        with patch.object(brainscore_vision, 'model_registry',
                          {'test-legacy': lambda: legacy}):
            with patch('brainscore_vision.import_plugin'):
                model = brainscore_vision.load_model('test-legacy')

        assert isinstance(model, UnifiedModel)
        assert isinstance(model, VisionModelAdapter)
        assert model.identifier == 'test-legacy'

    def test_load_model_wrapped_legacy_interact_scores(self):
        import brainscore_vision
        expected = _vision_neural_assembly()
        legacy = _make_legacy_model(
            identifier='test-legacy',
            region_layer_map={'IT': 'layer4'},
        )
        legacy.look_at.return_value = expected

        with patch.object(brainscore_vision, 'model_registry',
                          {'test-legacy': lambda: legacy}):
            with patch('brainscore_vision.import_plugin'):
                model = brainscore_vision.load_model('test-legacy')

        scored = score_stimuli(model, _vision_stimulus_set(), record='IT')

        assert isinstance(model, VisionModelAdapter)
        xr.testing.assert_identical(scored, expected)

    def test_load_model_does_not_double_wrap_unified(self):
        """If the model is already a UnifiedModel, don't wrap it."""
        import brainscore_vision

        native = BrainScoreModel(
            identifier='native-model',
            model=None,
            region_layer_map={},
            preprocessors={'vision': lambda m, s, **kw: None},
        )

        with patch.object(brainscore_vision, 'model_registry',
                          {'native-model': lambda: native}):
            with patch('brainscore_vision.import_plugin'):
                model = brainscore_vision.load_model('native-model')

        assert model is native
        assert not isinstance(model, VisionModelAdapter)
