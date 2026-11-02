import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import MagicMock, PropertyMock, patch

from brainscore_core.model_interface import TaskContext, UnifiedModel, BrainScoreModel
from brainscore_core.streaming_helpers import neural_response
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly, NeuroidAssembly
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

    def test_neural_response_participate_matches_legacy_look_at_exactly(self):
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
        scored = neural_response(stream_adapter, stimuli, record='IT')

        xr.testing.assert_identical(scored, legacy_output)
        legacy_stream.start_recording.assert_called_once_with(
            'IT', [(70, 170)]
        )

    def test_participate_is_one_look_at_with_the_session_table_and_settings(self):
        from brainscore_core.streaming import BufferedSession
        stimuli = _vision_stimulus_set()
        legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
        session = BufferedSession(stimulus_set=stimuli,
                                  requested_output_channels=('neural:IT',),
                                  time_bins=[(50, 100)])
        session.settings = {'number_of_trials': 3}
        VisionModelAdapter(legacy).participate(session)
        legacy.start_recording.assert_called_once_with('IT', [(50, 100)])
        legacy.look_at.assert_called_once()
        args, kwargs = legacy.look_at.call_args
        assert args[0] is stimuli and args[1] == 3 and kwargs == {}
        event, = session.emitted
        assert event.payload == 'mock_assembly' and event.meta == {'driver': 'legacy'}

    def test_participate_projects_undeclared_input_columns(self):
        from brainscore_core.streaming import BufferedSession
        stimuli = _vision_stimulus_set()
        stimuli['sentence'] = ['a cat', 'a dog']
        legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
        session = BufferedSession(stimulus_set=stimuli,
                                  requested_output_channels=('neural:IT',))
        session.settings = {'number_of_trials': 2, 'require_variance': True}
        VisionModelAdapter(legacy).participate(session)
        args, kwargs = legacy.look_at.call_args
        assert legacy.look_at.call_count == 1
        assert 'sentence' not in args[0].columns and 'sentence' in stimuli.columns
        assert list(args[0]['stimulus_id']) == ['s0', 's1']
        assert args[1] == 2 and kwargs == {'require_variance': True}

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

    def test_start_task_without_fitting_stimuli_passes_only_the_task(self):
        """No fitting data means no fitting argument, as the legacy benchmark called it.

        The real OddOneOut.start_task(self, task) rejects an extra positional None;
        TestStrictLegacyStartTask proves this against that exact signature.
        """
        legacy = _make_legacy_model()
        adapter = VisionModelAdapter(legacy)
        ctx = TaskContext(task_type='label')

        adapter.start_task(ctx)

        legacy.start_task.assert_called_once_with('label')

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


class _StrictOddOneOut:
    """The real OddOneOut signatures: no *args or fitting slot to absorb an invented None."""
    identifier = 'strict-odd-one-out'

    def __init__(self):
        self.tasks, self.looks = [], []
        self.output = BehavioralAssembly(np.array(['cat', 'dog']), dims=['presentation'])

    def visual_degrees(self):
        return 8

    def start_task(self, task):
        self.tasks.append(task)

    def look_at(self, stimuli, number_of_trials=1):
        self.looks.append((stimuli, number_of_trials))
        return self.output


class _StrictTaskWithSettings(_StrictOddOneOut):
    """A fitting-free task that still takes start_task settings by keyword."""

    def start_task(self, task, number_of_trials=1, require_variance=False):
        self.tasks.append((task, number_of_trials, require_variance))


class _StrictProbabilities(_StrictOddOneOut):
    """ProbabilitiesMapping.start_task: fitting data is a required positional argument."""

    def start_task(self, task, fitting_stimuli, number_of_trials=1, require_variance=False):
        self.tasks.append((task, fitting_stimuli, number_of_trials, require_variance))


class TestStrictLegacyStartTask:
    """Hebart2023 calls start_task(BrainModel.Task.odd_one_out) with no fitting data."""

    def test_direct_task_reaches_a_task_only_signature(self):
        legacy = _StrictOddOneOut()
        VisionModelAdapter(legacy).start_task('odd_one_out')
        assert legacy.tasks == ['odd_one_out']

    def test_task_context_reaches_a_task_only_signature(self):
        legacy = _StrictOddOneOut()
        VisionModelAdapter(legacy).start_task(TaskContext(task_type='odd_one_out'))
        assert legacy.tasks == ['odd_one_out']

    @pytest.mark.parametrize('form', ['direct', 'context'])
    def test_settings_without_fitting_are_kept(self, form):
        legacy = _StrictTaskWithSettings()
        adapter = VisionModelAdapter(legacy)
        settings = {'number_of_trials': 2, 'require_variance': True}
        if form == 'direct':
            adapter.start_task('odd_one_out', **settings)
        else:
            adapter.start_task(TaskContext(task_type='odd_one_out',
                                           metadata={'task_settings': settings}))
        assert legacy.tasks == [('odd_one_out', 2, True)]

    def test_fitting_data_is_still_passed_positionally(self):
        legacy = _StrictProbabilities()
        fitting = _vision_stimulus_set()
        VisionModelAdapter(legacy).start_task('probabilities', fitting,
                                              number_of_trials=2, require_variance=True)
        assert len(legacy.tasks) == 1
        task, seen, trials, variance = legacy.tasks[0]
        assert (task, trials, variance) == ('probabilities', 2, True) and seen is fitting

    def test_facade_session_runs_the_task_once_for_repeated_measurements(self):
        from brainscore_core.legacy_facade import LegacyBenchmarkFacade
        legacy = _StrictOddOneOut()
        facade = LegacyBenchmarkFacade(VisionModelAdapter(legacy), visual_degrees=8)
        first, second = _vision_stimulus_set(), _vision_stimulus_set()

        facade.start_task('odd_one_out')
        outputs = [facade.look_at(first), facade.look_at(second, number_of_trials=3)]

        assert legacy.tasks == ['odd_one_out']
        assert len(legacy.looks) == 2
        assert all(seen is sent for (seen, _), sent in zip(legacy.looks, (first, second)))
        assert [trials for _, trials in legacy.looks] == [1, 3]
        assert all(output is legacy.output for output in outputs)

    def test_a_failed_configuration_is_retried(self):
        class _FailsOnce(_StrictOddOneOut):
            def start_task(self, task):
                super().start_task(task)
                if len(self.tasks) == 1:
                    raise RuntimeError('first configuration fails')

        legacy = _FailsOnce()
        adapter = VisionModelAdapter(legacy)
        with pytest.raises(RuntimeError):
            adapter.start_task('odd_one_out')
        adapter.start_task('odd_one_out')
        adapter.start_task('odd_one_out')
        assert legacy.tasks == ['odd_one_out', 'odd_one_out']


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

    def test_load_model_wrapped_legacy_participate_scores(self):
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

        scored = neural_response(model, _vision_stimulus_set(), record='IT')

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


@pytest.mark.parametrize('use_session', [False, True], ids=['process', 'session'])
def test_real_wrapper_matches_legacy_recording(tmp_path, monkeypatch, use_session):
    """Exercise Core and Vision together without downloaded weights or data."""
    import torch
    from PIL import Image

    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
    from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    paths = {}
    for index, value in enumerate([1, 2]):
        path = tmp_path / f'image-{index}.png'
        Image.new('RGB', (2, 2), (value, value, value)).save(path)
        paths[f's{index}'] = path
    stimuli = StimulusSet({'stimulus_id': list(paths)})
    stimuli.identifier = 'adapter-runtime-images'
    stimuli.stimulus_paths = paths

    def preprocess(image_paths):
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1))
        return np.stack(images)

    network = torch.nn.Sequential(torch.nn.Conv2d(3, 1, kernel_size=2, bias=False))
    with torch.no_grad():
        network[0].weight.fill_(1)
    wrapper = PytorchWrapper(
        model=network,
        preprocessing=preprocess,
        identifier='adapter-runtime-model',
        batch_size=1,
    )
    legacy = ModelCommitment(
        identifier='adapter-runtime-model',
        activations_model=wrapper,
        layers=['0'],
        region_layer_map={'IT': '0'},
    )
    legacy.start_recording('IT', time_bins=[(70, 170)])
    expected = legacy.look_at(stimuli, number_of_trials=1)
    np.testing.assert_array_equal(expected.values.reshape(-1), [12, 24])
    legacy.reset()

    adapter = VisionModelAdapter(legacy)
    if use_session:
        actual = neural_response(adapter, stimuli, record='IT')
    else:
        adapter.start_recording('IT', time_bins=[(70, 170)])
        actual = adapter.process(stimuli)

    xr.testing.assert_identical(actual, expected)
    assert list(actual['stimulus_id'].values) == ['s0', 's1']
    assert not network[0]._forward_hooks
    adapter.reset()


@pytest.mark.parametrize('workflow_name', [
    'map_new_plugins.yml',
    'automerge_plugin-only_prs.yml',
    'check_if_pr_is_automergeable.yml',
])
def test_v2_prs_require_manual_submission_actions(workflow_name):
    from pathlib import Path
    import yaml

    workflows = Path(__file__).resolve().parents[1] / '.github/workflows'
    workflow = yaml.load(
        (workflows / workflow_name).read_text(),
        Loader=yaml.BaseLoader,
    )
    assert workflow['on']['pull_request']['branches-ignore'] == [
        'unified-model-interface-v2',
    ]
    assert 'branches' not in workflow['on']['pull_request']

    # The normal v2 checks must still run for PRs and pushes.
    checks = yaml.load((workflows / 'umi-v2.yml').read_text(), Loader=yaml.BaseLoader)
    for event in ['pull_request', 'push']:
        assert checks['on'][event]['branches'] == ['unified-model-interface-v2']
