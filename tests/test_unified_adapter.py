import pytest
from unittest.mock import MagicMock, PropertyMock

from brainscore_core.model_interface import TaskContext, UnifiedModel
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


class TestVisionAdapterStartTask:

    def test_start_task_unwraps_context(self):
        """Legacy start_task(task, fitting_stimuli) takes TWO args."""
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
