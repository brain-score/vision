from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

import brainscore_vision
from brainscore_core import Score, Benchmark
from brainscore_vision import BrainModel, StimulusSet
# Note that we cannot import `from brainscore_vision.model_helpers.generic_plugin_tests import test_*` directly
# since this would expose the `test_*` methods during pytest test collection
from brainscore_vision.benchmark_helpers import generic_plugin_tests
from brainscore_vision.benchmark_helpers.screen import place_on_screen


class TestIdentifier:
    BenchmarkClass = namedtuple("DummyBenchmark", field_names=['identifier'])

    def test_proper_identifier(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkClass(identifier='dummy')
        generic_plugin_tests.test_identifier('dummy')

    def test_no_identifier_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkClass(identifier=None)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_identifier(identifier='dummy')


class TestCeiling:
    BenchmarkClass = namedtuple("DummyBenchmark", field_names=['ceiling'])

    def test_proper_ceiling(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkClass(ceiling=Score(.42))
        generic_plugin_tests.test_ceiling('dummy')

    def test_no_ceiling_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkClass(ceiling=None)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_ceiling('dummy')

    def test_negative_ceiling_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkClass(ceiling=Score(-0.1))
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_ceiling('dummy')

    def test_nan_ceiling_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkClass(ceiling=Score(np.nan))
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_ceiling('dummy')

    def test_above_1_ceiling_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkClass(ceiling=Score(1.1))
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_ceiling('dummy')


class TestCallsStartTaskOrRecording:
    class BenchmarkDummy(Benchmark):
        def __init__(self, start_task=False, start_recording=False):
            self.start_task = start_task
            self.start_recording = start_recording

        def __call__(self, candidate: BrainModel) -> Score:
            if self.start_task:
                candidate.start_task(BrainModel.Task.passive)
            if self.start_recording:
                candidate.start_recording(recording_target=BrainModel.RecordingTarget.V1, time_bins=[(50, 100)])

    def test_calls_start_task(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(start_task=True)
        generic_plugin_tests.test_calls_start_task_or_recording('dummy')

    def test_calls_start_recording(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(start_recording=True)
        generic_plugin_tests.test_calls_start_task_or_recording('dummy')

    def test_calls_neither_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(start_task=False, start_recording=False)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_calls_start_task_or_recording('dummy')


class TestStartsValidTask:
    class BenchmarkDummy(Benchmark):
        def __init__(self, task: BrainModel.Task):
            self.task = task

        def __call__(self, candidate: BrainModel) -> Score:
            candidate.start_task(self.task)

    @pytest.mark.parametrize('task', [
        BrainModel.Task.passive, BrainModel.Task.label, BrainModel.Task.probabilities, BrainModel.Task.odd_one_out])
    def test_valid_task(self, mocker, task):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task=task)
        generic_plugin_tests.TestStartTask().test_starts_valid_task('dummy')

    def test_None_task_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task=None)
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartTask().test_starts_valid_task('dummy')

    def test_unregistered_task_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task='takeovertheplanet')
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartTask().test_starts_valid_task('dummy')

    def test_not_calling_start_task_skips(self, mocker):
        class BenchmarkDummy(Benchmark):
            def __call__(self, candidate: BrainModel) -> Score:
                candidate.start_recording(recording_target=BrainModel.RecordingTarget.V1, time_bins=[(50, 100)])

        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = BenchmarkDummy()
        generic_plugin_tests.TestStartTask().test_starts_valid_task('dummy')


class TestTaskValidFittingStimuli:
    class BenchmarkDummy(Benchmark):
        def __init__(self, task: BrainModel.Task, fitting_stimuli):
            self.task = task
            self.fitting_stimuli = fitting_stimuli

        def __call__(self, candidate: BrainModel) -> Score:
            candidate.start_task(self.task, fitting_stimuli=self.fitting_stimuli)

    # passive

    def test_passive_None(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task=BrainModel.Task.passive, fitting_stimuli=None)
        generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    def test_passive_with_stimuli_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task=BrainModel.Task.passive, fitting_stimuli=StimulusSet(
            {'stimulus_id': [1, 2, 3], 'image_label': [1, 2, 3]}))
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    # oddoneout

    def test_oddoneout_None(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task=BrainModel.Task.odd_one_out, fitting_stimuli=None)
        generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    def test_oddoneout_with_stimuli_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task=BrainModel.Task.odd_one_out, fitting_stimuli=StimulusSet(
            {'stimulus_id': [1, 2, 3], 'image_label': [1, 2, 3]}))
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    # label

    def test_label_imagenet_descriptor(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task=BrainModel.Task.label, fitting_stimuli='imagenet')
        generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    def test_label_other_descriptor_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task=BrainModel.Task.label, fitting_stimuli='objectnet')
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    def test_label_list_of_labels(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task=BrainModel.Task.label, fitting_stimuli=['dog', 'cat'])
        generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    def test_label_list_of_labels_int_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task=BrainModel.Task.label, fitting_stimuli=[1, 2])
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    def test_label_None_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(task=BrainModel.Task.label, fitting_stimuli=None)
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    # probabilities

    def test_probabilities_fitting_stimuli(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        stimulus_set = StimulusSet({'stimulus_id': [1, 2, 3], 'image_label': [1, 2, 3]})
        load_mock.return_value = self.BenchmarkDummy(
            task=BrainModel.Task.probabilities, fitting_stimuli=_add_stimulus_set_paths(stimulus_set))
        generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    def test_probabilities_fitting_stimuli_without_stimulus_id_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        stimulus_set = StimulusSet({'stimulus_num': [1, 2, 3], 'image_label': [1, 2, 3]})
        load_mock.return_value = self.BenchmarkDummy(
            task=BrainModel.Task.probabilities, fitting_stimuli=stimulus_set)
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    def test_probabilities_fitting_stimuli_without_image_label_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        stimulus_set = StimulusSet({'stimulus_id': [1, 2, 3]})
        load_mock.return_value = self.BenchmarkDummy(
            task=BrainModel.Task.probabilities, fitting_stimuli=_add_stimulus_set_paths(stimulus_set))
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')


def _add_stimulus_set_paths(stimulus_set: StimulusSet) -> StimulusSet:
    paths = [Path(__file__).parent / 'rgb1.jpg', Path(__file__).parent / 'rgb1-10to12.png']
    stimulus_set.stimulus_paths = {stimulus_id: paths[num % len(paths)]
                                   for num, stimulus_id in enumerate(stimulus_set['stimulus_id'])}
    return stimulus_set


class TestTestStartRecordingValidTarget:
    class BenchmarkDummy(Benchmark):
        def __init__(self, recording_target):
            self.recording_target = recording_target

        def __call__(self, candidate: BrainModel) -> Score:
            candidate.start_recording(self.recording_target, time_bins=None)

    @pytest.mark.parametrize('recording_target', [
        BrainModel.RecordingTarget.V1,
        BrainModel.RecordingTarget.V2,
        BrainModel.RecordingTarget.V4,
        BrainModel.RecordingTarget.IT,
    ])
    def test_valid_target(self, recording_target, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(recording_target=recording_target)
        generic_plugin_tests.TestStartRecording().test_starts_valid_recording_target('dummy')

    def test_None_target_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(recording_target=None)
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartRecording().test_starts_valid_recording_target('dummy')

    def test_invalid_string_target_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(recording_target='V0')
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartRecording().test_starts_valid_recording_target('dummy')

    def test_int_target_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(recording_target=1)
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartRecording().test_starts_valid_recording_target('dummy')


class TestTestStartRecordingValidTimebins:
    class BenchmarkDummy(Benchmark):
        def __init__(self, time_bins):
            self.time_bins = time_bins

        def __call__(self, candidate: BrainModel) -> Score:
            candidate.start_recording(BrainModel.RecordingTarget.IT, time_bins=self.time_bins)

    @pytest.mark.parametrize('time_bins', [
        [(70, 170)],
        [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70)],
        [(50, 70), (60, 80), (70, 90)],
        [(100, 150), (100, 120), (100, 101)],
        [(100, 500)],
        [(np.int64(70), np.int64(170))],
    ])
    def test_valid_timebins(self, time_bins, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(time_bins=time_bins)
        generic_plugin_tests.TestStartRecording().test_starts_valid_recording_timebins('dummy')

    @pytest.mark.parametrize('time_bins', [
        [(70, 70)],
        [(80, 70)],
        [(-10, 0)],
    ])
    def test_invalid_timebins_fails(self, time_bins, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(time_bins=time_bins)
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartRecording().test_starts_valid_recording_timebins('dummy')

    def test_None_timebins_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(time_bins=None)
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartRecording().test_starts_valid_recording_timebins('dummy')


class TestCallsModelLookAt:
    class BenchmarkDummy(Benchmark):
        def __init__(self, call_look_at: bool):
            self.call_look_at = call_look_at

        def __call__(self, candidate: BrainModel):
            stimulus_set = StimulusSet({'stimulus_id': [1, 2, 3], 'image_label': [1, 2, 3]})
            if self.call_look_at:
                candidate.look_at(stimulus_set)

    def test_proper_call(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(call_look_at=True)
        generic_plugin_tests.test_calls_model_look_at('dummy')

    def test_not_calling_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(call_look_at=False)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_calls_model_look_at('dummy')


class TestTakesIntoAccountModelVisualDegrees:
    class BenchmarkDummy(Benchmark):
        def __init__(self, do_place_on_screen: bool, use_candidate_visual_degrees: bool, parent: str = 'neural'):
            self.do_place_on_screen = do_place_on_screen
            self.use_candidate_visual_degrees = use_candidate_visual_degrees
            self._parent = parent

        def __call__(self, candidate: BrainModel):
            stimulus_set = StimulusSet({'stimulus_id': [1, 2, 3], 'image_label': [1, 2, 3]})
            stimulus_set.identifier = 'dummy_stimulusset'
            stimulus_set = _add_stimulus_set_paths(stimulus_set)
            target_degrees = 3 if not self.use_candidate_visual_degrees else candidate.visual_degrees()
            if self.do_place_on_screen:
                stimulus_set = place_on_screen(stimulus_set=stimulus_set, source_visual_degrees=4,
                                               target_visual_degrees=target_degrees)
            candidate.look_at(stimulus_set)

        @property
        def parent(self) -> str:
            return self._parent

    def test_properly_uses_place_on_screen(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(do_place_on_screen=True, use_candidate_visual_degrees=True)
        generic_plugin_tests.test_takesintoaccount_model_visual_degrees('dummy')

    def test_not_using_place_on_screen_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(do_place_on_screen=False, use_candidate_visual_degrees=True)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_takesintoaccount_model_visual_degrees('dummy')

    def test_not_using_visual_degrees_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(do_place_on_screen=True, use_candidate_visual_degrees=False)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_takesintoaccount_model_visual_degrees('dummy')

    @pytest.mark.parametrize('parent', [
        'engineering',
        'Geirhos2021-top1',
        'ImageNet-top1',
        'ImageNet-C-top1',
    ])
    def test_not_using_either_passes_for_engineering(self, parent, mocker):  # these tests are expected to skip
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(do_place_on_screen=False, use_candidate_visual_degrees=False,
                                                     parent=parent)
        generic_plugin_tests.test_takesintoaccount_model_visual_degrees('dummy')


class TestValidScore:
    class BenchmarkDummy(Benchmark):
        def __init__(self, score: Score):
            self.score = score

        def __call__(self, candidate: BrainModel):
            return self.score

    @pytest.mark.parametrize('score', [
        Score(0.5),
        Score(1.0),
        Score(1),
        Score(0.0),
        Score(0),
        Score(0.123456789),
        Score(np.nan),  # explicitly allowed (signaling that the model outputs are not compliant)
        .14,
    ])
    def test_valid(self, score: Score, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(score=score)
        generic_plugin_tests.test_valid_score('dummy')

    def test_valid_with_attrs(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        score = Score(0.42)
        score.attrs['ceiling'] = Score(0.8)
        raw = Score(0.336)
        raw.attrs['raw'] = Score([0.3, 0.4, 0.4, 0.3, 0.5], coords={'split': [1, 2, 3, 4, 5]}, dims=['split'])
        score.attrs['raw'] = raw
        score.attrs['benchmark_identifier'] = 'my_benchmark'
        score.attrs['model_identifier'] = 'my_model'
        load_mock.return_value = self.BenchmarkDummy(score=score)
        generic_plugin_tests.test_valid_score('dummy')

    @pytest.mark.parametrize('score', [
        Score(1.0000000000001),
        Score(1.1),
        Score(2),
        Score(-0.1),
        Score(-1),
        None,
        '12',
        'score',
        (0.6, 0.3),
        np.array([0.12, 0.1]),
    ])
    def test_invalid_fails(self, score, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkDummy(score=score)
        with pytest.raises((AssertionError, TypeError, ValueError)):
            generic_plugin_tests.test_valid_score('dummy')


@pytest.mark.slow
@pytest.mark.private_access
@pytest.mark.parametrize('plugin_directory', [
    'rajalingham2020', 'geirhos2021', 'hebart2023',
    'majajhong2015',
    'imagenet'
])
def test_existing_benchmark_plugin(plugin_directory):
    command = [
        generic_plugin_tests.__file__,
        "--plugin_directory", Path(brainscore_vision.__file__).parent / 'benchmarks' / plugin_directory
    ]
    retcode = pytest.main(command)
    assert retcode == 0, "Tests failed"  # https://docs.pytest.org/en/latest/reference/exit-codes.html
