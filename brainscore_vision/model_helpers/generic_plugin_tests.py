from pathlib import Path

from brainio.stimuli import StimulusSet
# the following import is needed to configure pytest
# noinspection PyUnresolvedReferences
from brainscore_core.plugin_management.generic_plugin_tests_helper import pytest_generate_tests
from brainscore_vision import BrainModel, load_model
import pytest


def test_identifier(identifier: str):
    model = load_model(identifier)
    assert model.identifier is not None


def test_visual_degrees(identifier: str):
    model = load_model(identifier)
    assert 0 < model.visual_degrees() < 9000


def test_start_task_or_recording(identifier: str):
    model = load_model(identifier)
    can_do = ProbeModel()
    assert can_do.can_start_task(model) or can_do.can_start_recording(model)


def test_look_at_behavior_probabilities(identifier: str):
    model = load_model(identifier)
    stimuli = fitting_stimuli = _make_stimulus_set()
    if not ProbeModel().can_start_task_specific(model,
                                                task=BrainModel.Task.probabilities, fitting_stimuli=fitting_stimuli):
        # model cannot do this task, ignore. We're testing for behavior or neural in `test_supports_behavior_or_neural`
        return

    model.start_task(BrainModel.Task.probabilities, fitting_stimuli=stimuli)
    predictions = model.look_at(stimuli=stimuli, number_of_trials=1)
    assert set(predictions.dims) == {'presentation', 'choice'}
    assert set(predictions['stimulus_id'].values) == {'stimid1', 'stimid2', 'stimid3'}
    assert all(predictions['object_name'] == 'rgb')
    assert set(predictions['choice'].values) == set(fitting_stimuli['image_label'].values)
    assert (0 <= predictions.values).all()
    assert (predictions.values <= 1).all()

@pytest.mark.memory_intense
def test_look_at_neural_V1(identifier: str):
    model = load_model(identifier)
    if not ProbeModel().can_start_recording_region(model, recording_target=BrainModel.RecordingTarget.V1):
        # model cannot do this task, ignore. We're testing for behavior or neural in `test_supports_behavior_or_neural`
        return

    stimuli = _make_stimulus_set()
    model.start_recording(recording_target=BrainModel.RecordingTarget.V1, time_bins=[(50, 100)])
    predictions = model.look_at(stimuli=stimuli, number_of_trials=1)

    assert set(predictions['stimulus_id'].values) == {'stimid1', 'stimid2', 'stimid3'}
    assert all(predictions['object_name'] == 'rgb')
    assert len(predictions['neuroid']) >= 1, "expected at least one neuroid"
    assert len(set(predictions['neuroid_id'].values)) == len(predictions['neuroid']), "expected unique neuroid_ids"
    assert len(predictions.dims) <= 3
    assert 'presentation' in predictions.dims
    assert 'neuroid' in predictions.dims
    if len(predictions.dims) == 3:
        assert 'time_bin' in predictions.dims


def _make_stimulus_set() -> StimulusSet:
    stimuli = StimulusSet({
        'stimulus_id': ['stimid1', 'stimid2', 'stimid3'],
        'object_name': ['rgb', 'rgb', 'rgb'],
        'image_label': ['label1', 'label1', 'label2'],
        'filename': ['rgb1', 'rgb2', 'rgb3'],
    })
    stimuli.stimulus_paths = {'stimid1': Path(__file__).parent / 'generic_plugin_tests_resources' / 'rgb1.jpg',
                              'stimid2': Path(__file__).parent / 'generic_plugin_tests_resources' / 'rgb2.jpg',
                              'stimid3': Path(__file__).parent / 'generic_plugin_tests_resources' / 'rgb3.png',
                              }
    stimuli.identifier = 'test_look_at_neural_V1.rgb_1_2'
    return stimuli


class ProbeModel:
    def can_start_task(self, model: BrainModel) -> bool:
        tasks = [BrainModel.Task.label,
                 BrainModel.Task.probabilities,
                 BrainModel.Task.odd_one_out]
        for task in tasks:
            if self.can_start_task_specific(model, task=task):
                return True
        # no task worked
        return False

    def can_start_task_specific(self, model, task: BrainModel.Task, fitting_stimuli=None) -> bool:
        try:  # start task without fitting stimuli
            model.start_task(task=task)
            return True
        except Exception:
            try:  # start task with fitting stimuli
                model.start_task(task=task, fitting_stimuli=fitting_stimuli)
                return True
            except Exception:
                return False

    def can_start_recording(self, model: BrainModel) -> bool:
        regions = [BrainModel.RecordingTarget.V1,
                   BrainModel.RecordingTarget.V2,
                   BrainModel.RecordingTarget.V4,
                   BrainModel.RecordingTarget.IT]
        for region in regions:
            if self.can_start_recording_region(model, recording_target=region):
                return True
        # no region worked
        return False

    def can_start_recording_region(self, model: BrainModel, recording_target: BrainModel.RecordingTarget) -> bool:
        try:
            model.start_recording(recording_target=recording_target, time_bins=[(100, 200)])
            return True
        except Exception:
            return False
