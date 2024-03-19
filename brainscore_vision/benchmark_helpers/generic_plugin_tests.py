from pathlib import Path
from typing import List, Tuple, Union
from numpy.random import RandomState
import pytest

from brainio.assemblies import BehavioralAssembly, NeuroidAssembly
from brainio.stimuli import StimulusSet
from brainscore_core import Benchmark
from brainscore_core import Score
# the following import is needed to configure pytest
# noinspection PyUnresolvedReferences
from brainscore_core.plugin_management.generic_plugin_tests_helper import pytest_generate_tests
from brainscore_vision import load_benchmark, BrainModel


class ProbeModel(BrainModel):
    """
    Probing class to run a benchmark only until a specified point and keep track of call parameters.
    """

    class STOP:
        start_task = 'start_task'
        start_recording = 'start_recording'
        look_at = 'look_at'

    # stop_on: Union[ProbeModel.STOP, List[ProbeModel.STOP]]
    def __init__(self, stop_on, visual_degrees: int = 8):
        self._stop_on = stop_on
        self.stopped_on = None

        self.task = None
        self.fitting_stimuli = None
        self.recording_target = None
        self.time_bins = None
        self.visual_degrees_called = False
        self.look_at_stimuli = None

        self._visual_degrees = visual_degrees

    def start_task(self, task: BrainModel.Task, fitting_stimuli=None):
        self.task = task
        self.fitting_stimuli = fitting_stimuli
        if self._stop_on == ProbeModel.STOP.start_task or ProbeModel.STOP.start_task in self._stop_on:
            self.stopped_on = ProbeModel.STOP.start_task
            raise StopIteration("planned stop")

    def start_recording(self, recording_target: BrainModel.RecordingTarget, time_bins: List[Tuple[int]]):
        self.recording_target = recording_target
        self.time_bins = time_bins
        if self._stop_on == ProbeModel.STOP.start_recording or ProbeModel.STOP.start_recording in self._stop_on:
            self.stopped_on = ProbeModel.STOP.start_recording
            raise StopIteration("planned stop")

    def look_at(self, stimuli: Union[StimulusSet, List[str]], number_of_trials=1) \
            -> Union[BehavioralAssembly, NeuroidAssembly]:
        self.look_at_stimuli = stimuli
        if self._stop_on == ProbeModel.STOP.look_at or ProbeModel.STOP.look_at in self._stop_on:
            self.stopped_on = ProbeModel.STOP.look_at
            raise StopIteration("planned stop")
        return self._simulate_output(stimuli)

    def _simulate_output(self, stimuli: StimulusSet) -> Union[BehavioralAssembly, NeuroidAssembly]:
        stimulus_coords = {column: ('presentation', stimuli[column]) for column in stimuli.columns}
        rnd = RandomState(1)

        if self.task is not None and self.task is not BrainModel.Task.passive:  # behavior
            if self.task == BrainModel.Task.label:
                if self.fitting_stimuli == 'imagenet':
                    some_imagenet_synsets = ['n02107574', 'n02123045', 'n02804414']
                    choices = rnd.choice(some_imagenet_synsets, size=len(stimuli), replace=True)
                else:
                    choices = rnd.choice(self.fitting_stimuli, size=len(stimuli), replace=True)
                return BehavioralAssembly([choices], coords=stimulus_coords, dims=['choice', 'presentation'])
            elif self.task == BrainModel.Task.probabilities:
                labels = list(sorted(set(self.fitting_stimuli['image_label'])))
                probabilities = rnd.uniform(low=0, high=1, size=(len(stimuli), len(labels)))
                probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)  # have choices sum to 1
                return BehavioralAssembly(probabilities, coords={**stimulus_coords, **{'choice': ('choice', labels)}},
                                          dims=['presentation', 'choice'])
            elif self.task == BrainModel.Task.odd_one_out:
                triplet_choices = [0, 1, 2]
                choices = rnd.choice(triplet_choices, size=len(stimuli) // 3, replace=True)
                stimulus_ids = stimuli['stimulus_id'].values
                return BehavioralAssembly([choices], coords={
                    'stimulus_id': ('presentation', ['|'.join([f"{stimulus_ids[offset + i]}" for i in range(3)])
                                                     for offset in range(0, len(stimulus_ids) - 2, 3)]),
                }, dims=['choice', 'presentation'])
            else:
                return pytest.skip("task not implemented in probe")

        elif self.recording_target is not None:  # neural
            num_units = 300
            activity = rnd.random(size=(len(stimuli), num_units, self.time_bins))
            return NeuroidAssembly(activity, coords={**stimulus_coords, **{
                'neuroid_id': ('neuroid', np.arange(num_units)),
                'region': ('neuroid', [self.recording_target] * num_units),
                'time_bin_start': ('time_bin', [start for start, end in self.time_bins]),
                'time_bin_end': ('time_bin', [end for start, end in self.time_bins]),
            }}, dims=['presentation', 'neuroid', 'time_bin'])

        else:
            raise ValueError("no task or recording set")

    def visual_degrees(self) -> int:
        self.visual_degrees_called = True
        return self._visual_degrees


def test_identifier(identifier: str):
    benchmark = load_benchmark(identifier)
    assert benchmark.identifier is not None


def test_ceiling(identifier: str):
    benchmark = load_benchmark(identifier)
    assert benchmark.ceiling is not None
    assert 0 <= benchmark.ceiling <= 1


def test_calls_start_task_or_recording(identifier: str):
    benchmark = load_benchmark(identifier)
    probe_model = ProbeModel(stop_on=(ProbeModel.STOP.start_task, ProbeModel.STOP.start_recording))
    _run_with_stop(benchmark, probe_model)
    assert (probe_model.task is not None) or (probe_model.recording_target is not None)


class TestStartTask:
    def test_starts_valid_task(self, identifier: str):
        benchmark = load_benchmark(identifier)
        probe_model = ProbeModel(stop_on=(ProbeModel.STOP.start_task, ProbeModel.STOP.start_recording))
        _run_with_stop(benchmark, probe_model)
        if not probe_model.stopped_on == ProbeModel.STOP.start_task:
            return pytest.skip("benchmark does not call start_task")
        # at this point we know that the start_task method was called
        all_tasks = [attribute for attribute in dir(BrainModel.Task) if not attribute.startswith('_')]
        assert probe_model.task in all_tasks

    def test_task_valid_fitting_stimuli(self, identifier: str):
        benchmark = load_benchmark(identifier)
        probe_model = ProbeModel(stop_on=(ProbeModel.STOP.start_task, ProbeModel.STOP.start_recording))
        _run_with_stop(benchmark, probe_model)
        if not probe_model.stopped_on == ProbeModel.STOP.start_task:
            return pytest.skip("benchmark does not call start_task")
        # at this point we know that the start_task method was called

        # for the second argument, there are different options depending on the task
        if probe_model.task == BrainModel.Task.passive:
            assert probe_model.fitting_stimuli is None

        elif probe_model.task == BrainModel.Task.label:
            assert probe_model.fitting_stimuli is not None
            if isinstance(probe_model.fitting_stimuli, str):  # string specification of a bag of labels
                assert probe_model.fitting_stimuli == 'imagenet'  # only possible choice at the moment
            else:  # not a string, has to be a list of labels
                assert all(isinstance(label, str) for label in probe_model.fitting_stimuli), \
                    "every list item should be a string"


        elif probe_model.task == BrainModel.Task.probabilities:
            assert probe_model.fitting_stimuli is not None
            assert 'stimulus_id' in probe_model.fitting_stimuli
            stimulus_paths = [probe_model.fitting_stimuli.get_stimulus(stimulus_id)
                              for stimulus_id in probe_model.fitting_stimuli['stimulus_id']]
            assert all(Path(path).is_file() for path in stimulus_paths)
            assert 'image_label' in probe_model.fitting_stimuli

        elif probe_model.task == BrainModel.Task.odd_one_out:
            assert probe_model.fitting_stimuli is None


class TestStartRecording:
    def test_starts_valid_recording_target(self, identifier: str):
        benchmark = load_benchmark(identifier)
        probe_model = ProbeModel(stop_on=(ProbeModel.STOP.start_task, ProbeModel.STOP.start_recording))
        _run_with_stop(benchmark, probe_model)
        if not probe_model.stopped_on == ProbeModel.STOP.start_recording:
            return pytest.skip("benchmark does not call start_recording")
        # at this point we know that the start_recording method was called
        all_recording_targets = [attribute for attribute in dir(BrainModel.RecordingTarget)
                                 if not attribute.startswith('_')]
        assert probe_model.recording_target in all_recording_targets

    def test_starts_valid_recording_timebins(self, identifier: str):
        benchmark = load_benchmark(identifier)
        probe_model = ProbeModel(stop_on=(ProbeModel.STOP.start_task, ProbeModel.STOP.start_recording))
        _run_with_stop(benchmark, probe_model)
        if not probe_model.stopped_on == ProbeModel.STOP.start_recording:
            return pytest.skip("benchmark does not call start_recording")
        # at this point we know that the start_recording method was called
        assert probe_model.time_bins is not None
        for time_bin_num, (time_bin_start, time_bin_stop) in enumerate(probe_model.time_bins):
            assert isinstance(time_bin_start, int), f"time_bin {time_bin_num} start is not an integer: {time_bin_start}"
            assert isinstance(time_bin_stop, int), f"time_bin {time_bin_num} stop is not an integer: {time_bin_stop}"
            assert time_bin_start >= 0, f"time_bin {time_bin_num} start is < 0: {time_bin_start}"
            assert time_bin_start < time_bin_stop, (f"time_bin {time_bin_num} start is not before stop: "
                                                    f"({time_bin_start}, {time_bin_stop})")


def test_calls_model_look_at(identifier: str):
    benchmark = load_benchmark(identifier)
    probe_model = ProbeModel(stop_on=ProbeModel.STOP.look_at)
    _run_with_stop(benchmark, probe_model)  # will raise on its own if model.look_at isn't called


def test_takesintoaccount_model_visual_degrees(identifier: str):
    # make sure place_on_screen is called by adding a marker in that method
    benchmark = load_benchmark(identifier)
    probe_model = ProbeModel(stop_on=ProbeModel.STOP.look_at)
    _run_with_stop(benchmark, probe_model)
    assert (hasattr(probe_model.look_at_stimuli, 'placed_on_screen')
            and probe_model.look_at_stimuli.placed_on_screen is True), \
        ("Benchmark should place stimuli on screen using the `place_on_screen` method "
         "to take the visual degrees of the experiment and the (computational) subject into account")
    assert probe_model.visual_degrees_called, "Benchmark should use models' visual_degrees to place stimuli on screen"


class TestScore:
    def test_valid_behavioral_score(self, identifier: str):
        benchmark = load_benchmark(identifier)
        probe_model = ProbeModel(stop_on=[])
        score = benchmark(probe_model)
        self._validate_score(score)

    def test_valid_neural_score(self, identifier: str):
        ...

    def _validate_score(self, score: Score):
        assert 0 <= score <= 1


def _run_with_stop(benchmark: Benchmark, model: ProbeModel):
    try:
        benchmark(model)  # model will raise stop at some point
        assert False, "model did not stop on its own"
    except StopIteration:
        return
