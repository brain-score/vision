import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import MagicMock, PropertyMock, patch

from brainscore_core.contract import TaskContext, Subject
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
    legacy.look_at.return_value = _vision_neural_assembly()
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


class TestVisionAdapterIsASubject:

    def test_isinstance(self):
        # score() wraps a model that is not a Subject; the adapter must not be wrapped again.
        from brainscore_core.contract import Subject
        adapter = VisionModelAdapter(_make_legacy_model())
        assert isinstance(adapter, Subject)
        assert type(adapter).__bases__ == (Subject,)    # no other base class


class TestVisionAdapterIdentity:

    def test_identifier_from_property(self):
        legacy = _make_legacy_model(identifier='cornet-s')
        adapter = VisionModelAdapter(legacy)
        assert adapter.identifier == 'cornet-s'

    def test_declares_image_channel_only(self):
        adapter = VisionModelAdapter(_make_legacy_model())
        assert adapter.in_channels == {'vision:image'}
        assert adapter.required_channels == {'vision:image'}

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
        scored = neural_response(stream_adapter, stimuli, ['neural:IT'])

        xr.testing.assert_identical(scored, legacy_output)
        legacy_stream.start_recording.assert_called_once_with(
            'IT', [(70, 170)]
        )

    def test_participate_is_one_look_at_with_the_session_table_and_settings(self):
        from brainscore_core.streaming_helpers import stimulus_session
        stimuli = _vision_stimulus_set()
        legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
        session = stimulus_session(stimuli, ('neural:IT',), time_bins=[(50, 100)])
        session.settings = {'number_of_trials': 3}
        VisionModelAdapter(legacy).participate(session)
        legacy.start_recording.assert_called_once_with('IT', [(50, 100)])
        legacy.look_at.assert_called_once()
        args, kwargs = legacy.look_at.call_args
        assert args[0] is stimuli and args[1] == 3 and kwargs == {}
        event, = session.emitted
        assert event.payload is legacy.look_at.return_value and event.meta == {'driver': 'legacy'}

    def test_several_regions_in_one_session_make_the_calls_of_one_session_per_region(self):
        from unittest.mock import call
        from brainscore_core.streaming_helpers import stimulus_session
        stimuli = _vision_stimulus_set()
        outputs = {'V4': _vision_neural_assembly(), 'IT': _vision_neural_assembly() + 1}

        def run(channels_per_session):
            legacy = _make_legacy_model(region_layer_map={'V4': 'layer3', 'IT': 'layer4'})
            region = {}
            legacy.start_recording.side_effect = lambda target, bins: region.update(now=target)
            legacy.look_at.side_effect = lambda stimuli, trials: outputs[region['now']]
            adapter, emitted = VisionModelAdapter(legacy), []
            for channels in channels_per_session:
                session = stimulus_session(stimuli, channels, time_bins=[(50, 100)])
                adapter.participate(session)
                emitted += session.emitted
            return legacy.mock_calls, emitted

        joint_calls, joint = run([('neural:V4', 'neural:IT')])
        alone_calls, alone = run([('neural:V4',), ('neural:IT',)])
        assert joint_calls == alone_calls
        assert [c for c in joint_calls if c[0] in ('start_recording', 'look_at')][::2] == [
            call.start_recording('V4', [(50, 100)]), call.start_recording('IT', [(50, 100)])]
        assert [(e.channel, e.t_ms, e.meta) for e in joint] == [(e.channel, e.t_ms, e.meta) for e in alone]
        assert all(a.payload is b.payload for a, b in zip(joint, alone))

    def test_behavior_and_a_region_in_one_session_are_refused(self):
        from brainscore_core.streaming_helpers import stimulus_session
        legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
        session = stimulus_session(_vision_stimulus_set(), ('behavior', 'neural:IT'))
        from brainscore_core.compatibility import CompatibilityError, ExecutionNotFeasibleError
        with pytest.raises(ExecutionNotFeasibleError, match='one output channel per look_at') as caught:
            VisionModelAdapter(legacy).participate(session)
        assert caught.value.channels == {'behavior', 'neural:IT'}
        assert not isinstance(caught.value, CompatibilityError)
        legacy.look_at.assert_not_called()

    def test_participate_projects_undeclared_input_columns(self):
        from brainscore_core.streaming_helpers import stimulus_session
        stimuli = _vision_stimulus_set()
        stimuli['sentence'] = ['a cat', 'a dog']
        legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
        session = stimulus_session(stimuli, ('neural:IT',))
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

    def test_the_configured_fitting_set_is_kept_alive_so_its_id_is_not_reused(self):
        """The setup key holds id(fitting); a freed set's id could match a new set."""
        import gc
        import weakref

        class Fitting:
            pass

        class Legacy:                        # keeps no reference to what it is given
            identifier = 'legacy'

            def start_task(self, task, fitting_stimuli=None, **settings):
                pass

        adapter = VisionModelAdapter(Legacy())
        fitting = Fitting()
        alive = weakref.ref(fitting)
        adapter.start_task('probabilities', fitting)
        del fitting
        gc.collect()
        assert alive() is not None


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

    def test_task_reaches_a_task_only_signature(self):
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

        assert isinstance(model, Subject)
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

        scored = neural_response(model, _vision_stimulus_set(), ['neural:IT'])

        assert isinstance(model, VisionModelAdapter)
        xr.testing.assert_identical(scored, expected)

    def test_load_model_does_not_double_wrap_unified(self):
        """If the model is already a Subject, do not wrap it."""
        import brainscore_vision

        class NativeSubject(Subject):
            identifier = 'native-subject'
            in_channels = set()
            out_channels = set()

            def participate(self, session):
                pass

            def reset(self):
                pass

        native = NativeSubject()

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
        actual = neural_response(adapter, stimuli, ['neural:IT'])
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


class _RecordingLegacyModel:
    """Legacy BrainModel stand-in that logs calls and accepts any time_bins, as TemporalAligned does."""
    identifier = 'recording-legacy'

    def __init__(self):
        self.calls = []
        self.layer_model = type('L', (), {'region_layer_map': {'IT': 'layer4'}})()

    def visual_degrees(self):
        return 8

    def start_task(self, task, *args, **kwargs):
        self.calls.append(('start_task', task))

    def start_recording(self, target, time_bins):
        self.calls.append(('start_recording', target, time_bins))

    def look_at(self, stimuli, number_of_trials=1, require_variance=False):
        self.calls.append(('look_at',))
        return _vision_neural_assembly()


class TestTimeBinsFidelity:
    """The facade hands the legacy model the benchmark's time bins unchanged."""

    TEMPORAL_BINS = np.array([(t, t + 10) for t in range(0, 300, 10)], dtype=object)

    def _through_facade(self, time_bins):
        from brainscore_core.legacy_facade import LegacyBenchmarkFacade
        legacy = _RecordingLegacyModel()
        facade = LegacyBenchmarkFacade(VisionModelAdapter(legacy), visual_degrees=8)
        facade.start_recording('IT', time_bins=time_bins)
        facade.look_at(_vision_stimulus_set())
        return legacy

    def test_ndarray_of_several_bins_matches_the_plain_legacy_call(self):
        plain = _RecordingLegacyModel()
        plain.start_recording('IT', self.TEMPORAL_BINS)
        legacy = self._through_facade(self.TEMPORAL_BINS)
        (_, target, sent), = [c for c in legacy.calls if c[0] == 'start_recording']
        assert target == 'IT' and sent is self.TEMPORAL_BINS
        assert legacy.calls[-1] == ('look_at',)

    def test_ndarray_bins_are_applied_once_per_configuration(self):
        legacy = _RecordingLegacyModel()
        adapter = VisionModelAdapter(legacy)
        adapter.start_recording('IT', time_bins=self.TEMPORAL_BINS)
        adapter.start_recording('IT', time_bins=self.TEMPORAL_BINS.copy())
        assert len(legacy.calls) == 1

    def test_list_bins_are_passed_unchanged(self):
        bins = [(50, 100), (100, 150)]
        legacy = self._through_facade(bins)
        assert legacy.calls[0] == ('start_recording', 'IT', bins)

    @pytest.mark.parametrize('empty', [None, [], np.array([])])
    def test_missing_bins_keep_the_legacy_default(self, empty):
        legacy = self._through_facade(empty)
        assert legacy.calls[0] == ('start_recording', 'IT', [(70, 170)])


    def test_container_change_keeps_the_base_setup_call_count(self):
        from brainscore_core.legacy_facade import LegacyBenchmarkFacade
        legacy = _RecordingLegacyModel()
        facade = LegacyBenchmarkFacade(VisionModelAdapter(legacy), visual_degrees=8)
        for bins in [((70, 170),), [(70, 170)]]:
            facade.start_recording('IT', time_bins=bins)
            facade.look_at(_vision_stimulus_set())
        assert len([c for c in legacy.calls if c[0] == 'start_recording']) == 2

    def test_list_key_matches_base_exactly(self):
        bins = [(70.0000000001, 170), (170, 270)]
        adapter = VisionModelAdapter(_RecordingLegacyModel())
        adapter.start_recording('IT', time_bins=bins)
        assert adapter._configured_recording == (repr('IT'), repr(bins))

    @pytest.mark.parametrize('dtype', [float, object])
    def test_distinct_ndarray_bins_never_share_a_setup_key(self, dtype):
        first = np.array([[70., 170.], [170., 270.]], dtype=dtype)
        second = first.copy()
        second[0, 0] = 70.000000001
        if dtype is float:
            assert repr(list(first)) == repr(list(second))  # equal display, distinct bins
        legacy = _RecordingLegacyModel()
        adapter = VisionModelAdapter(legacy)
        adapter.start_recording('IT', time_bins=first)
        adapter.start_recording('IT', time_bins=second)
        assert len(legacy.calls) == 2 and legacy.calls[-1][2] is second

class _CommitmentLike:
    """ModelCommitment.start_task / look_at: any task but 'passive' turns behavior on."""
    identifier = 'commitment-like'

    def __init__(self):
        self.do_behavior = False
        self.calls = []
        self.layer_model = type('L', (), {'region_layer_map': {'IT': 'layer4'}})()

    def visual_degrees(self):
        return 8

    def start_task(self, task, *args, **kwargs):
        self.calls.append(('start_task', task))
        self.do_behavior = task != 'passive'

    def start_recording(self, target, time_bins):
        self.calls.append(('start_recording', target))

    def look_at(self, stimuli, number_of_trials=1, require_variance=False):
        self.calls.append(('look_at',))
        if self.do_behavior:
            return BehavioralAssembly(np.array(['cat', 'dog']), dims=['presentation'],
                                      coords={'stimulus_id': ('presentation', ['s0', 's1'])})
        return _vision_neural_assembly()


class TestBehaviorDoesNotLeakIntoNeuralSessions:

    def _task(self):
        stimuli = _vision_stimulus_set()
        return TaskContext(task_type='probabilities', fitting_stimuli=stimuli)

    def test_behavior_then_neural_returns_neural(self):
        from brainscore_core.streaming_helpers import behavioral_response
        legacy = _CommitmentLike()
        adapter = VisionModelAdapter(legacy)
        assert isinstance(behavioral_response(adapter, _vision_stimulus_set(), self._task()), BehavioralAssembly)
        neural = neural_response(adapter, _vision_stimulus_set(), ['neural:IT'])
        assert isinstance(neural, NeuroidAssembly)
        assert legacy.calls[-3:] == [('start_task', 'passive'), ('start_recording', 'IT'), ('look_at',)]

    def test_neural_on_a_fresh_adapter_makes_the_same_legacy_calls(self):
        legacy = _CommitmentLike()
        neural_response(VisionModelAdapter(legacy), _vision_stimulus_set(), ['neural:IT'])
        assert legacy.calls == [('start_recording', 'IT'), ('look_at',)]

    def test_repeated_neural_sessions_send_passive_at_most_once(self):
        from brainscore_core.streaming_helpers import behavioral_response
        legacy = _CommitmentLike()
        adapter = VisionModelAdapter(legacy)
        behavioral_response(adapter, _vision_stimulus_set(), self._task())
        for _ in range(2):
            neural_response(adapter, _vision_stimulus_set(), ['neural:IT'])
        assert legacy.calls.count(('start_task', 'passive')) == 1

    def test_two_legacy_facades_in_sequence(self):
        from brainscore_core.legacy_facade import LegacyBenchmarkFacade
        legacy = _CommitmentLike()
        adapter = VisionModelAdapter(legacy)
        stimuli = _vision_stimulus_set()
        behavioral = LegacyBenchmarkFacade(adapter, visual_degrees=8)
        behavioral.start_task('odd_one_out')
        choices = behavioral.look_at(stimuli)
        assert isinstance(choices, BehavioralAssembly)
        assert list(choices.values) == ['cat', 'dog']          # the legacy model's answer
        neural = LegacyBenchmarkFacade(adapter, visual_degrees=8)
        neural.start_recording('IT', time_bins=[(70, 170)])
        activity = neural.look_at(stimuli)
        assert isinstance(activity, NeuroidAssembly)
        np.testing.assert_array_equal(activity.values, _vision_neural_assembly().values)
        assert legacy.calls[-3:] == [('start_task', 'passive'), ('start_recording', 'IT'), ('look_at',)]

    def test_benchmark_that_returns_to_passive_matches_the_plain_legacy_model(self):
        from brainscore_core.legacy_facade import LegacyBenchmarkFacade
        stimuli = _vision_stimulus_set()

        def run(model):
            model.start_task('odd_one_out')
            model.look_at(stimuli)
            model.start_task('passive')
            model.start_recording('IT', time_bins=[(70, 170)])
            return model.look_at(stimuli)

        plain = _CommitmentLike()
        expected = run(plain)
        legacy = _CommitmentLike()
        actual = run(LegacyBenchmarkFacade(VisionModelAdapter(legacy), visual_degrees=8))
        assert isinstance(expected, NeuroidAssembly) and isinstance(actual, NeuroidAssembly)
        assert legacy.calls == plain.calls


class _LegacyForcedChoice:
    """A legacy model whose probabilities task gives rows over its own column order."""
    identifier = 'legacy-forced-choice'

    def __init__(self):
        self.tasks = []

    def start_task(self, task, fitting_stimuli=None, **kwargs):
        self.tasks.append(task)

    def look_at(self, stimuli, number_of_trials=1):
        ids = list(stimuli['stimulus_id'])
        values = np.array([[0.2, 0.8], [0.7, 0.3], [0.5, 0.5]][:len(ids)])
        return BehavioralAssembly(values, dims=['presentation', 'choice'], coords={
            'stimulus_id': ('presentation', ids), 'position': ('presentation', list(range(len(ids)))),
            'choice': ['pseudo', 'real']})

    def visual_degrees(self):
        return 8


def _choice_stimuli():
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b', 'c'],
                                        'image_file_name': ['a.png', 'b.png', 'c.png'],
                                        'image_label': ['real', 'pseudo', 'real']}))
    stimuli.identifier = 'adapter-choice'
    stimuli.stimulus_paths = dict(zip(stimuli['stimulus_id'], stimuli['image_file_name']))
    return stimuli


class TestVisionAdapterChoice:
    """A Choice task on behavior:choice: the legacy probabilities task, argmax in its own order."""

    def test_declares_the_choice_and_keeps_bare_behavior(self):
        adapter = VisionModelAdapter(_LegacyForcedChoice())
        assert {'behavior', 'behavior:choice', 'behavior:probabilities'} <= adapter.out_channels

    def test_a_choice_task_is_answered_from_the_legacy_probabilities_task(self):
        from brainscore_core.responses import Choice
        from brainscore_core.streaming_helpers import stimulus_session
        legacy = _LegacyForcedChoice()
        task = TaskContext(response=Choice(['real', 'pseudo']), fitting_stimuli=_choice_stimuli())
        session = stimulus_session(_choice_stimuli(), ['behavior:choice'], task)
        VisionModelAdapter(legacy).participate(session)
        choice = session.collect('behavior:choice')
        assert legacy.tasks == ['probabilities']
        assert list(choice['choice'].values) == ['real', 'pseudo']
        # argmax in the legacy column order: a tie (row c) goes to its first column, 'pseudo'
        np.testing.assert_array_equal(choice.values, [[1, 0], [0, 1], [0, 1]])
        record = choice.attrs['response']
        assert (record['origin'], record['method'], record['conversion'], record['parameters']) == (
            'derived', 'legacy', 'argmax', {'tie': 'first', 'option_order': 'legacy'})
        assert list(choice['raw_answer'].values) == ['[0.2, 0.8]', '[0.7, 0.3]', '[0.5, 0.5]']

    def test_a_choice_task_without_fitting_stimuli_runs_the_label_task(self):
        """Without fitting stimuli the legacy label task answers (native): one-hot rows over
        the task's labels, every presentation coordinate of the legacy answer kept."""
        from brainscore_core.responses import Choice
        from brainscore_core.streaming_helpers import stimulus_session
        legacy = _LegacyLabels(['real', 'pseudo', 'real'])
        session = stimulus_session(_choice_stimuli(), ['behavior:choice'],
                                   TaskContext(response=Choice(['real', 'pseudo'])))
        VisionModelAdapter(legacy).participate(session)
        choice = session.collect('behavior:choice')
        assert legacy.tasks == [('label', ['real', 'pseudo'])]
        np.testing.assert_array_equal(choice.values, [[1, 0], [0, 1], [1, 0]])
        assert list(choice['raw_answer'].values) == ['real', 'pseudo', 'real']
        assert list(choice['position'].values) == [0, 1, 2]
        record = choice.attrs['response']
        assert (record['origin'], record['method'], record['conversion'], record['parameters']) == (
            'native', 'legacy', None, {})

    def test_the_probabilities_task_answers_natively_in_float64_in_the_task_order(self):
        from brainscore_core.responses import Choice
        from brainscore_core.streaming_helpers import stimulus_session
        legacy = _LegacyForcedChoice()
        task = TaskContext(response=Choice(['real', 'pseudo']), fitting_stimuli=_choice_stimuli())
        session = stimulus_session(_choice_stimuli(), ['behavior:probabilities'], task)
        VisionModelAdapter(legacy).participate(session)
        rows = session.collect('behavior:probabilities')
        assert legacy.tasks == ['probabilities']
        assert list(rows['choice'].values) == ['real', 'pseudo']          # the task's order
        assert rows.values.dtype == np.float64
        assert rows.values.tolist() == [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]
        record = rows.attrs['response']
        assert (record['origin'], record['method'], record['parameters']['option_order']) == (
            'native', 'legacy', 'task')
        assert list(rows['raw_answer'].values) == ['[0.2, 0.8]', '[0.7, 0.3]', '[0.5, 0.5]']

    def test_probabilities_for_other_labels_than_the_fitting_set_are_refused_before_set_up(self):
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        from brainscore_core.responses import Choice
        from brainscore_core.streaming_helpers import stimulus_session
        legacy = _LegacyForcedChoice()
        task = TaskContext(response=Choice(['real', 'fake']), fitting_stimuli=_choice_stimuli())
        session = stimulus_session(_choice_stimuli(), ['behavior:probabilities'], task)
        with pytest.raises(ExecutionNotFeasibleError, match=r"fitted on the labels \['pseudo', 'real'\]"):
            VisionModelAdapter(legacy).participate(session)
        assert legacy.tasks == [] and session.emitted == []


class _LegacyLabels(_LegacyForcedChoice):
    """A legacy model whose label task answers one label per stimulus, dims (choice, presentation)."""
    identifier = 'legacy-labels'

    def __init__(self, answers):
        super().__init__()
        self.answers = answers

    def start_task(self, task, fitting_stimuli=None, **kwargs):
        self.tasks.append((task, list(fitting_stimuli)) if task == 'label' else task)

    def look_at(self, stimuli, number_of_trials=1):
        ids = list(stimuli['stimulus_id'])
        return BehavioralAssembly([self.answers[:len(ids)]], dims=['choice', 'presentation'], coords={
            'stimulus_id': ('presentation', ids), 'position': ('presentation', list(range(len(ids))))})


class _LegacyWithoutProbabilities(_LegacyForcedChoice):
    """A legacy BrainModel that is not a ModelCommitment and has no probabilities task."""
    identifier = 'legacy-label-only'

    def start_task(self, task, fitting_stimuli=None, **kwargs):
        if task != 'label':
            raise NotImplementedError(f'unsupported task {task}')
        self.tasks.append(task)

    def look_at(self, stimuli, number_of_trials=1):
        raise AssertionError('never called')


class _FileFeatures:
    """An offline extractor: one feature per image, read from the image file itself."""

    def __init__(self):
        self.shown = []

    def __call__(self, stimuli, layers, number_of_trials=1, require_variance=False):
        values = []
        for stimulus_id in stimuli['stimulus_id']:
            with open(stimuli.get_stimulus(stimulus_id), 'rb') as image:
                values.append([float(len(image.read()))])
            self.shown.append(stimulus_id)
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
        return NeuroidAssembly(np.array(values), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
            'image_label': ('presentation', list(stimuli['image_label'])),
            'neuroid_id': ('neuroid', [0]), 'layer': ('neuroid', ['L'])})


def _image_set(folder, prefix, missing=()):
    ids = [f'{prefix}{i}' for i in range(4)]
    paths = {sid: str(folder / f'{sid}.png') for sid in ids}
    for sid, path in paths.items():
        if sid not in missing:
            with open(path, 'wb') as image:
                image.write(b'x' * (10 + ids.index(sid)))
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ids, 'image_file_name': list(paths.values()),
                                        'image_label': ['real', 'pseudo'] * 2}))
    stimuli.identifier = f'adapter-choice-{prefix}'
    stimuli.stimulus_paths = paths
    return stimuli


def _choice_session(test, fitting):
    """A core session for a Choice task with fitting stimuli, as ROAR builds it."""
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    return stimulus_session(test, ['behavior:choice'], TaskContext(
        response=Choice(['real', 'pseudo']), fitting_stimuli=fitting))


def test_a_missing_fitting_file_passes_through_as_file_not_found(tmp_path):
    """An operational error of the readout fit is not turned into a refusal (core only;
    the same through brainscore.score is in unified tests/test_legacy_adapters_on_roar.py)."""
    from brainscore_vision.model_helpers.brain_transformation.behavior import ProbabilitiesMapping
    features = _FileFeatures()
    legacy = ProbabilitiesMapping('legacy-probabilities', features, 'L')
    session = _choice_session(_image_set(tmp_path, 't'), _image_set(tmp_path, 'f', missing={'f2'}))
    with pytest.raises(FileNotFoundError, match='f2.png'):
        VisionModelAdapter(legacy).participate(session)
    assert features.shown == ['f0', 'f1']                  # no test stimulus is shown
    assert session.emitted == []                           # nothing is stored


def test_a_legacy_model_that_refuses_probabilities_raises_its_own_error(tmp_path):
    legacy = _LegacyWithoutProbabilities()
    session = _choice_session(_image_set(tmp_path, 't'), _image_set(tmp_path, 'f'))
    with pytest.raises(NotImplementedError, match='unsupported task probabilities'):
        VisionModelAdapter(legacy).participate(session)
    assert legacy.tasks == [] and session.emitted == []


def test_fitting_stimuli_without_labels_give_an_error_with_a_remedy():
    """A probabilities task fitted on a table with no label column is refused before set-up."""
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    legacy = _LegacyForcedChoice()
    task = TaskContext(response=Choice(['real', 'pseudo']),
                       fitting_stimuli=_choice_stimuli().drop(columns='image_label'))
    session = stimulus_session(_choice_stimuli(), ['behavior:probabilities'], task)
    with pytest.raises(ValueError, match="'image_label'"):
        VisionModelAdapter(legacy).participate(session)
    assert legacy.tasks == [] and session.emitted == []


class _RecordsLookAt(_LegacyLabels):
    """A legacy model that keeps the arguments of each look_at."""

    def __init__(self, answers):
        super().__init__(answers)
        self.looks = []

    def look_at(self, stimuli, number_of_trials=1, require_variance=False):
        self.looks.append({'number_of_trials': number_of_trials, 'require_variance': require_variance})
        if self.tasks and self.tasks[-1] == 'probabilities':
            return _LegacyForcedChoice.look_at(self, stimuli)
        return super().look_at(stimuli, number_of_trials)


@pytest.mark.parametrize('channel, fitting', [('behavior:choice', None),          # the label route
                                             ('behavior:choice', True),          # the fitted choice
                                             ('behavior:probabilities', True)])
def test_the_choice_routes_forward_the_trial_count_and_the_variance_request(channel, fitting):
    """S9-03: as the bare route does, the format routes forward both measurement settings."""
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    legacy = _RecordsLookAt(['real', 'pseudo', 'real'])
    task = TaskContext(response=Choice(['real', 'pseudo']),
                       fitting_stimuli=_choice_stimuli() if fitting else None)
    session = stimulus_session(_choice_stimuli(), [channel], task,
                               settings={'number_of_trials': 3, 'require_variance': True})
    VisionModelAdapter(legacy).participate(session)
    assert legacy.looks == [{'number_of_trials': 3, 'require_variance': True}]


def _twenty_stimuli(identifier):
    ids = [f's{i}' for i in range(20)]
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ids, 'image_file_name': [f'{i}.png' for i in ids],
                                        'image_label': ['real'] * 20}))
    stimuli.identifier = identifier
    stimuli.stimulus_paths = dict(zip(ids, stimuli['image_file_name']))
    return stimuli


class _OutsideLabels(_LegacyLabels):
    """An old model whose label task answers 'real' except for n_outside labels outside the
    list, or the given ``answers`` (an array of one label per stimulus). Its answer carries
    every column of the stimulus set and an attribute of its own, as an old answer can."""

    def __init__(self, n_outside=0, answers=None):
        super().__init__([])
        self.n_outside = n_outside
        self.answers = answers

    def look_at(self, stimuli, number_of_trials=1, require_variance=False):
        n = len(stimuli)
        answers = self.answers if self.answers is not None else \
            ['real'] * (n - self.n_outside) + ['unlisted'] * self.n_outside
        return BehavioralAssembly([answers], dims=['choice', 'presentation'], coords={
            column: ('presentation', list(stimuli[column])) for column in stimuli.columns},
            attrs={'old_model_note': 'kept'})


def _label_run(n_outside, looks=1):
    """The old model above, asked through the facade the score entry points give a legacy
    benchmark."""
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    facade = LegacyBenchmarkFacade(VisionModelAdapter(_OutsideLabels(n_outside)), visual_degrees=8)
    facade.start_task('label', ['real', 'pseudo'])
    for call in range(looks):
        facade.look_at(_twenty_stimuli(f'label-run-{n_outside}-{call}'))
    return facade


def test_a_label_outside_the_list_reaches_the_old_metric_unchanged_and_is_counted():
    """Owner decision 2026-11-16: no random label for an old model's label outside the list."""
    import warnings
    from brainscore_core.legacy_facade import OLD_LABEL_POLICY, record_replaced_answers
    from brainscore_core.metrics import Score
    # The adapter marks the row: a label outside the list is a NaN row.
    from brainscore_vision.compat.unified_adapter import _choice_from_labels
    answer = BehavioralAssembly([['real', 'unlisted']], dims=['choice', 'presentation'],
                                coords={'stimulus_id': ('presentation', ['a', 'b'])})
    rows = _choice_from_labels(answer, ['real', 'pseudo'])
    np.testing.assert_array_equal(rows.values[0], [1, 0])
    assert np.isnan(rows.values[1]).all()
    # Through the facade the row is not replaced; it is counted, and the count reaches the score.
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        facade = _label_run(1)
    assert facade.n_invalid_answers == 1
    score = Score(0.5)
    record_replaced_answers(score, facade)
    assert score.attrs['n_invalid'] == 1 and score.attrs['answer_policy'] == OLD_LABEL_POLICY


def _own_and_given(answers, labels, identifier):
    """The old model's own look_at answer and the facade's answer for the same stimuli."""
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    stimuli = _twenty_stimuli(identifier)
    own = _OutsideLabels(answers=answers).look_at(stimuli)
    facade = LegacyBenchmarkFacade(VisionModelAdapter(_OutsideLabels(answers=answers)), visual_degrees=8)
    facade.start_task('label', labels)
    return own, facade.look_at(stimuli), facade


_REAL = ['real'] * 15


@pytest.mark.parametrize('answers', [
    _REAL + ['unlisted'] * 5,                                    # str
    np.array(_REAL + ['unlisted'] * 5),                          # numpy str ('<U8')
    np.array([b'real'] * 15 + [b'unlisted'] * 5),                # bytes: never text, all outside
    np.array([1] * 15 + [7] * 5),                                # integers: all outside
], ids=['str', 'numpy-str', 'bytes', 'int'])
def test_the_facade_gives_the_old_benchmark_the_old_models_own_answer(answers):
    """The proof: with labels outside the list, the facade's answer equals the old model's own
    look_at answer in values, numpy dtype, dims, index level names, every presentation
    coordinate with its values and order, and attrs; the count on the score is true."""
    from brainscore_core.legacy_facade import OLD_LABEL_POLICY, record_replaced_answers
    from brainscore_core.metrics import Score
    labels = ['real', 'pseudo']
    own, given, facade = _own_and_given(answers, labels, f'own-answer-{type(answers[0]).__name__}')
    assert given.dims == own.dims
    assert given.dtype == own.dtype
    np.testing.assert_array_equal(given.values, own.values)
    assert list(given.indexes['presentation'].names) == list(own.indexes['presentation'].names)
    for name in own.indexes['presentation'].names:
        assert list(given[name].values) == list(own[name].values)
    assert given.attrs == own.attrs
    outside = sum(1 for value in np.asarray(own.values).reshape(-1)
                  if not any(value == label for label in labels))
    score = Score(0.5)
    record_replaced_answers(score, facade)
    assert score.attrs['n_invalid'] == outside > 0
    assert score.attrs['answer_policy'] == OLD_LABEL_POLICY


def test_the_integers_1_and_7_are_outside_the_options_1_and_7():
    """Inside the list means equal to a label: the integer 1 is not the text '1'."""
    answers = np.array([1, 7] + [1] * 18)
    own, given, facade = _own_and_given(answers, ['1', '7', 'other'], 'own-answer-integers')
    assert facade.n_invalid_answers == 20
    np.testing.assert_array_equal(given.values, own.values)
    assert given.dtype == own.dtype
    from brainscore_vision.compat.unified_adapter import _choice_from_labels
    two = BehavioralAssembly([[1, 7]], dims=['choice', 'presentation'],
                             coords={'stimulus_id': ('presentation', ['a', 'b'])})
    assert np.isnan(_choice_from_labels(two, ['1', '7']).values).all()   # two labels outside


@pytest.mark.parametrize('values, stimulus', [
    ([1.0, np.nan], 'b'),                                                    # float64 NaN
    (np.array([1.0, None], dtype=object), 'b'),                              # None
    (np.array([np.float32(np.nan), 1.0], dtype=object), 'a'),                # stored as a float NaN
    (np.array([np.float32(np.nan), np.int64(1)], dtype=object), 'a'),        # stored as a float NaN
], ids=['float64-nan', 'none', 'object-float32-nan-and-float', 'object-float32-nan-and-int64'])
def test_a_null_label_is_refused_before_anything_is_stored(values, stimulus):
    """A stated limit: the stored choice format reads a null raw answer (None, float NaN) as no
    answer, so the row would be dropped without a word; the adapter refuses it instead. The test
    reads the values as the stored answer holds them: the coordinate turns a float32 NaN in an
    object array into a float NaN."""
    from brainscore_vision.compat.unified_adapter import _choice_from_labels
    answer = BehavioralAssembly([values], dims=['choice', 'presentation'],
                                coords={'stimulus_id': ('presentation', ['a', 'b'])})
    with pytest.raises(ValueError, match=rf"stimulus '{stimulus}' is .*: the stored choice format "
                                         r"cannot carry a null label .* must return a label for every row"):
        _choice_from_labels(answer, ['cat', 'dog'])


def test_a_float32_nan_that_the_stored_answer_keeps_passes_with_its_row():
    """A homogeneous float32 array keeps its NaN as a float32, which is not null for core: both
    rows reach the old benchmark and both count as outside."""
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    stimuli = _twenty_stimuli('float32-nan').iloc[:2]
    stimuli = StimulusSet(stimuli)
    stimuli.identifier = 'float32-nan-two'
    stimuli.stimulus_paths = dict(zip(stimuli['stimulus_id'], stimuli['image_file_name']))
    answers = np.array([np.nan, 1.0], dtype=np.float32)
    facade = LegacyBenchmarkFacade(VisionModelAdapter(_OutsideLabels(answers=answers)), visual_degrees=8)
    facade.start_task('label', ['cat', 'dog'])
    given = facade.look_at(stimuli)
    assert given.shape == (1, 2) and given.dtype == np.float32 and facade.n_invalid_answers == 2


def test_a_float_label_that_is_not_nan_passes_unchanged_and_counts_as_outside():
    answers = np.array([2.5] * 20)
    own, given, facade = _own_and_given(answers, ['cat', 'dog'], 'own-answer-floats')
    np.testing.assert_array_equal(given.values, own.values)
    assert given.dtype == own.dtype and facade.n_invalid_answers == 20


def test_the_width_of_a_padded_text_array_is_not_kept():
    """A stated limit: the facade gives equal values and the same kind of numpy type, but a
    padded fixed-width text array ('<U10' for shorter labels) comes back as the width of its
    longest label, because the presentation index keeps scalar values only."""
    answers = np.array(_REAL + ['unlisted'] * 5, dtype='<U10')
    own, given, _ = _own_and_given(answers, ['real', 'pseudo'], 'own-answer-padded')
    np.testing.assert_array_equal(given.values, own.values)
    assert own.dtype == np.dtype('<U10') and given.dtype == np.dtype('<U8')


def test_an_old_model_with_one_label_row_for_each_trial_is_refused():
    """A recorded limit, older than the label change: core's rule for stored answers allows a
    stimulus as many answer rows as the table has it, so an old model that returns one label
    row for each trial (several rows for one stimulus) is refused. No shipped old model does
    it: the standard label behavior returns one row for each stimulus."""
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade

    class _TrialRows(_LegacyLabels):
        def look_at(self, stimuli, number_of_trials=1, require_variance=False):
            ids = [i for i in stimuli['stimulus_id'] for _ in range(number_of_trials)]
            return BehavioralAssembly([['real'] * len(ids)], dims=['choice', 'presentation'], coords={
                'stimulus_id': ('presentation', ids),
                'trial': ('presentation', list(range(number_of_trials)) * len(stimuli))})

    facade = LegacyBenchmarkFacade(VisionModelAdapter(_TrialRows([])), visual_degrees=8)
    facade.start_task('label', ['real', 'pseudo'])
    # The message names the channel, the stimulus and how often the table has it.
    with pytest.raises(ValueError, match=r"'behavior:choice'.*stimulus 's0' is answered more often "
                                         r"than the table has it \(1\)"):
        facade.look_at(_twenty_stimuli('trial-rows'), number_of_trials=3)


def test_the_count_is_the_sum_over_several_looks_and_a_new_facade_starts_at_zero():
    facade = _label_run(1, looks=3)
    assert facade.n_invalid_answers == 3
    assert _label_run(0).n_invalid_answers == 0


def test_a_run_without_invalid_answers_writes_a_zero_on_the_score():
    from brainscore_core.legacy_facade import record_replaced_answers
    from brainscore_core.metrics import Score
    score = Score(0.5)
    record_replaced_answers(score, _label_run(0))
    assert score.attrs['n_invalid'] == 0


@pytest.mark.parametrize('n_outside', [3, 20])   # 15 and 100 percent of 20
def test_no_share_of_labels_outside_the_list_stops_an_old_models_run(n_outside):
    assert _label_run(n_outside).n_invalid_answers == n_outside


# --- the answer of an old benchmark carries the stimulus columns -----------------------

def _columns_stimuli():
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': ['a', 'b', 'c', 'd'], 'image_file_name': ['a.png', 'b.png', 'c.png', 'd.png'],
        'animal': ['cat', 'dog', 'cat', 'dog'], 'image_type': ['w', 'w', 'f', 'f']}))
    stimuli.identifier = 'adapter-columns'
    stimuli.stimulus_paths = dict(zip(stimuli['stimulus_id'], stimuli['image_file_name']))
    return stimuli


class _FullColumnLabels(_LegacyForcedChoice):
    """An old model whose label answer carries every column of the stimulus set, as in old Brain-Score."""
    identifier = 'legacy-full-columns'

    def look_at(self, stimuli, number_of_trials=1, require_variance=False):
        coords = {name: ('presentation', list(stimuli[name])) for name in stimuli.columns}
        return BehavioralAssembly([['cat', 'dog', 'cat', 'dog']], dims=['choice', 'presentation'], coords=coords)


def test_an_old_model_through_the_adapter_gets_the_same_answer_as_before_the_repair():
    """The facade adds columns an answer lacks; this answer lacks none, so it is the answer of the adapter
    turned back to labels, as before."""
    import xarray as xr
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade, _labels_from_choice
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    stimuli = _columns_stimuli()
    facade = LegacyBenchmarkFacade(VisionModelAdapter(_FullColumnLabels()), visual_degrees=8)
    facade.start_task('label', ['cat', 'dog'])
    session = stimulus_session(stimuli, ['behavior:choice'], TaskContext(response=Choice(['cat', 'dog'])))
    VisionModelAdapter(_FullColumnLabels()).participate(session)
    before = _labels_from_choice(session.collect('behavior:choice'))          # no stimuli: the old turn back
    xr.testing.assert_identical(facade.look_at(stimuli), before)
    assert list(before.indexes['presentation'].names) == list(stimuli.columns)


def test_old_metrics_read_the_stimulus_columns_of_an_answer_through_the_adapter():
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    facade = LegacyBenchmarkFacade(VisionModelAdapter(_FullColumnLabels()), visual_degrees=8)
    facade.start_task('label', ['cat', 'dog'])
    answer = facade.look_at(_columns_stimuli())
    scores = [float((answer.sel(animal=cat, image_type='w').values == cat).mean())
              for cat in sorted(set(answer['animal'].values))]      # as the Baker metric reads 'animal'
    assert scores == [1.0, 1.0]
    assert list(answer.sortby('stimulus_id')['stimulus_id'].values) == ['a', 'b', 'c', 'd']   # as BMD


def test_a_fitted_choice_session_stores_the_same_presentation_index_as_before():
    """The facade attaches columns; the stored answer of a new-style fitted choice session (as ROAR
    makes it) keeps its index: the rows' stimulus_id and the record levels only."""
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    task = TaskContext(response=Choice(['real', 'pseudo']), fitting_stimuli=_choice_stimuli())
    session = stimulus_session(_choice_stimuli(), ['behavior:choice'], task)
    VisionModelAdapter(_LegacyForcedChoice()).participate(session)
    choice = session.collect('behavior:choice')
    assert list(choice.indexes['presentation'].names) == [
        'stimulus_id', 'raw_answer', 'response_state', 'response_reason']


class TestOneSessionAtATime:
    """One adapter holds the task and recording state of one session (unified issue 81)."""

    @staticmethod
    def _session():
        from brainscore_core.streaming_helpers import stimulus_session
        return stimulus_session(_vision_stimulus_set(), ('neural:IT',))

    def test_a_second_session_during_the_first_is_refused(self):
        legacy = _make_legacy_model(identifier='busy-model', region_layer_map={'IT': 'layer4'})
        adapter = VisionModelAdapter(legacy)
        refusals = []

        def look_at_that_starts_a_second_session(*args, **kwargs):
            with pytest.raises(RuntimeError, match='one session at a time') as refusal:
                adapter.participate(self._session())
            refusals.append(str(refusal.value))
            return _vision_neural_assembly()

        legacy.look_at.side_effect = look_at_that_starts_a_second_session
        first = self._session()
        adapter.participate(first)
        assert 'busy-model' in refusals[0]
        assert len(first.emitted) == 1   # the refused call did not end the first session

    def test_a_session_from_another_thread_is_refused(self):
        import threading
        legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
        adapter = VisionModelAdapter(legacy)
        started, release, errors = threading.Event(), threading.Event(), []

        def slow_look_at(*args, **kwargs):
            started.set()
            release.wait(5)
            return _vision_neural_assembly()

        legacy.look_at.side_effect = slow_look_at
        first_session, outcome = self._session(), {}

        def run_first():
            adapter.participate(first_session)
            outcome['first'] = 'done'
        first = threading.Thread(target=run_first)
        first.start()
        assert started.wait(5)
        try:
            adapter.participate(self._session())
        except RuntimeError as error:
            errors.append(error)
        finally:
            release.set()
            first.join(5)
        assert errors and 'one session at a time' in str(errors[0])
        assert not first.is_alive()
        assert outcome == {'first': 'done'}       # the first session finished successfully
        assert len(first_session.emitted) == 1

    def test_a_new_session_runs_after_a_session_that_failed(self):
        legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
        adapter = VisionModelAdapter(legacy)
        legacy.look_at.side_effect = [ValueError('legacy failure'), _vision_neural_assembly()]
        with pytest.raises(ValueError, match='legacy failure'):
            adapter.participate(self._session())
        session = self._session()
        adapter.participate(session)
        assert len(session.emitted) == 1

    def test_two_sessions_in_sequence_run_as_before(self):
        legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
        adapter = VisionModelAdapter(legacy)
        sessions = [self._session(), self._session()]
        for session in sessions:
            adapter.participate(session)
        for session in sessions:
            assert len(session.emitted) == 1
        assert legacy.look_at.call_count == 2
        assert legacy.start_recording.call_count == 1    # the same setup is not sent again
        region, bins = legacy.start_recording.call_args.args
        assert region == 'IT'
        (time_bin,) = bins
        assert tuple(time_bin) == (70, 170)
