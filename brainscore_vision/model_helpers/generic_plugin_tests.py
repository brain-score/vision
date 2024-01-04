# the following import is needed to configure pytest
# noinspection PyUnresolvedReferences
from brainscore_core.plugin_management.generic_plugin_tests_helper import pytest_addoption, pytest_generate_tests
from brainscore_vision import BrainModel, load_model


def test_has_identifier(identifier: str):
    model = load_model(identifier)
    assert model.identifier is not None


def test_has_visual_degrees(identifier: str):
    model = load_model(identifier)
    assert 0 < model.visual_degrees() < 9000


def test_supports_behavior_or_neural(identifier: str):
    model = load_model(identifier)
    assert can_do_behavior(model) or can_do_neural(model)


def can_do_behavior(model: BrainModel) -> bool:
    tasks = [BrainModel.Task.label,
             BrainModel.Task.probabilities,
             BrainModel.Task.odd_one_out]
    for task in tasks:
        if can_do_behavior_task(model, task=task):
            return True
    # no task worked
    return False


def can_do_behavior_task(model, task: BrainModel.Task) -> bool:
    try:  # start task without fitting stimuli
        model.start_task(task=task)
        return True
    except Exception:
        try:  # start task with fitting stimuli
            model.start_task(task=task, fitting_stimuli=None)
            return True
        except Exception:
            return False


def can_do_neural(model: BrainModel) -> bool:
    regions = [BrainModel.RecordingTarget.V1,
               BrainModel.RecordingTarget.V2,
               BrainModel.RecordingTarget.V4,
               BrainModel.RecordingTarget.IT]
    for region in regions:
        if can_do_neural_region(model, recording_target=region):
            return True
    # no region worked
    return False


def can_do_neural_region(model: BrainModel, recording_target: BrainModel.RecordingTarget) -> bool:
    try:  # start recording
        model.start_recording(recording_target=recording_target, time_bins=[(100, 200)])
        return True
    except Exception:
        return False
