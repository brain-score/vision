from collections import namedtuple
from pathlib import Path

import pytest

import brainscore_vision
from brainscore_core import Score, Benchmark
from brainscore_vision import BrainModel, StimulusSet
# Note that we cannot import `from brainscore_vision.model_helpers.generic_plugin_tests import test_*` directly
# since this would expose the `test_*` methods during pytest test collection
from brainscore_vision.benchmark_helpers import generic_plugin_tests


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
            task=BrainModel.Task.probabilities, fitting_stimuli=self._add_stimulus_set_paths(stimulus_set))
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
            task=BrainModel.Task.probabilities, fitting_stimuli=self._add_stimulus_set_paths(stimulus_set))
        with pytest.raises(AssertionError):
            generic_plugin_tests.TestStartTask().test_task_valid_fitting_stimuli('dummy')

    def _add_stimulus_set_paths(self, stimulus_set: StimulusSet) -> StimulusSet:
        paths = [Path(__file__).parent / 'rgb1.jpg', Path(__file__).parent / 'rgb1-10to12.png']
        stimulus_set.stimulus_paths = {stimulus_id: paths[num % len(paths)]
                                       for num, stimulus_id in enumerate(stimulus_set['stimulus_id'])}
        return stimulus_set


@pytest.mark.slow
@pytest.mark.private_access
def test_existing_benchmark_plugin():
    command = [
        generic_plugin_tests.__file__,
        "--plugin_directory", Path(brainscore_vision.__file__).parent / 'benchmarks' / 'rajalingham2020'
    ]
    retcode = pytest.main(command)
    assert retcode == 0, "Tests failed"  # https://docs.pytest.org/en/latest/reference/exit-codes.html
