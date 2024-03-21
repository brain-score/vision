from collections import namedtuple
from pathlib import Path
from typing import List, Tuple

import pytest
from numpy.random import RandomState

import brainscore_vision
from brainio.assemblies import BehavioralAssembly, NeuroidAssembly
from brainio.stimuli import StimulusSet
from brainscore_vision import BrainModel
# Note that we cannot import `from brainscore_vision.model_helpers.generic_plugin_tests import test_*` directly
# since this would expose the `test_*` methods during pytest test collection
from brainscore_vision.model_helpers import generic_plugin_tests


class TestIdentifier:
    ModelClass = namedtuple("DummyModel", field_names=['identifier'])

    def test_no_identifier_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(identifier=None)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_identifier(identifier='dummy')

    def test_proper_identifier(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(identifier='dummy')
        generic_plugin_tests.test_identifier('dummy')


class TestVisualDegrees:
    ModelClass = namedtuple("DummyModel", field_names=['visual_degrees'])

    def test_proper_degrees(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(visual_degrees=lambda: 8)
        generic_plugin_tests.test_visual_degrees('dummy')

    def test_degrees_0_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(visual_degrees=lambda: 0)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_visual_degrees(identifier='dummy')

    def test_degrees_None_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(visual_degrees=lambda: None)
        with pytest.raises(TypeError):
            generic_plugin_tests.test_visual_degrees(identifier='dummy')

    def test_no_function_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = object()
        with pytest.raises(AttributeError):
            generic_plugin_tests.test_visual_degrees(identifier='dummy')


class TestStartTaskOrRecording:
    ModelClass = namedtuple("DummyModel", field_names=['start_task', 'start_recording'], defaults=[None, None])

    def test_supports_all(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(start_task=lambda task, fitting_stimuli=None: None,
                                                 start_recording=lambda recording_target, time_bins: None)
        generic_plugin_tests.test_start_task_or_recording('dummy')

    def test_supports_task_only(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(start_task=lambda task, fitting_stimuli=None: None)
        generic_plugin_tests.test_start_task_or_recording('dummy')

    def test_supports_task_label_only(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def start_task(task: BrainModel.Task, fitting_stimuli):
            if task != BrainModel.Task.label:
                raise NotImplementedError()

        load_mock.return_value = self.ModelClass(start_task=start_task)
        generic_plugin_tests.test_start_task_or_recording('dummy')

    def test_supports_recording_only(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(start_recording=lambda recording_target, time_bins: None)
        generic_plugin_tests.test_start_task_or_recording('dummy')

    def test_supports_recording_V1_only(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def start_recording(recording_target: BrainModel.RecordingTarget, time_bins):
            if recording_target != BrainModel.RecordingTarget.V1:
                raise NotImplementedError()

        load_mock.return_value = self.ModelClass(start_recording=start_recording)
        generic_plugin_tests.test_start_task_or_recording('dummy')

    def test_supports_recording_IT_only(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def start_recording(recording_target: BrainModel.RecordingTarget, time_bins):
            if recording_target != BrainModel.RecordingTarget.IT:
                raise NotImplementedError()

        load_mock.return_value = self.ModelClass(start_recording=start_recording)
        generic_plugin_tests.test_start_task_or_recording('dummy')

    def test_supports_None_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass()
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_start_task_or_recording('dummy')


class TestLookAtBehaviorProbabilities:
    class ModelMock(BrainModel):
        def __init__(self, assembly_factory):
            self.assembly_factory = assembly_factory
            self.fitting_stimuli = None

        def start_task(self, task: BrainModel.Task, fitting_stimuli=None):
            assert task == BrainModel.Task.probabilities
            assert fitting_stimuli is not None
            self.fitting_stimuli = fitting_stimuli

        def look_at(self, stimuli: StimulusSet, number_of_trials=1) -> BehavioralAssembly:
            return self.assembly_factory(stimuli, self.fitting_stimuli)

    def test_proper_format(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def assembly_maker(stimuli: StimulusSet, fitting_stimuli: StimulusSet) -> BehavioralAssembly:
            rng = RandomState(1)
            choices = list(sorted(set(fitting_stimuli['image_label'])))
            probabilities = rng.rand(len(stimuli), len(choices))
            return BehavioralAssembly(probabilities,
                                      coords={**{coord: ('presentation', stimuli[coord]) for coord in stimuli},
                                              **{'choice': ('choice', choices)}},
                                      dims=['presentation', 'choice'])

        load_mock.return_value = self.ModelMock(assembly_maker)
        generic_plugin_tests.test_look_at_behavior_probabilities('dummy')

    def test_None_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        load_mock.return_value = self.ModelMock(lambda stimuli, fitting_stimuli: None)
        with pytest.raises(AttributeError):
            generic_plugin_tests.test_look_at_behavior_probabilities('dummy')

    def test_no_stimulus_id_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def assembly_maker(stimuli: StimulusSet, fitting_stimuli: StimulusSet) -> BehavioralAssembly:
            rng = RandomState(1)
            choices = list(sorted(set(fitting_stimuli['image_label'])))
            probabilities = rng.rand(len(stimuli), len(choices))
            return BehavioralAssembly(probabilities,
                                      coords={**{coord: ('presentation', stimuli[coord]) for coord in stimuli
                                                 if coord != 'stimulus_id'},
                                              **{'choice': ('choice', choices)}},
                                      dims=['presentation', 'choice'])

        load_mock.return_value = self.ModelMock(assembly_maker)
        with pytest.raises(KeyError):
            generic_plugin_tests.test_look_at_behavior_probabilities('dummy')

    def test_no_choice_dim_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def assembly_maker(stimuli: StimulusSet, fitting_stimuli: StimulusSet) -> BehavioralAssembly:
            rng = RandomState(1)
            choices = list(sorted(set(fitting_stimuli['image_label'])))
            # improperly output labels instead of probabilities
            return BehavioralAssembly(rng.choice(choices, size=len(stimuli)),
                                      coords={coord: ('presentation', stimuli[coord]) for coord in stimuli
                                              if coord != 'stimulus_id'},
                                      dims=['presentation'])

        load_mock.return_value = self.ModelMock(assembly_maker)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_look_at_behavior_probabilities('dummy')


class TestLookAtNeuralV1:
    class ModelMock(BrainModel):
        def __init__(self, assembly_factory):
            self.assembly_factory = assembly_factory
            self.time_bins = None

        def start_recording(self, recording_target: BrainModel.RecordingTarget, time_bins: List[Tuple[int, int]]):
            assert recording_target == BrainModel.RecordingTarget.V1
            assert len(time_bins) == 1
            self.time_bins = time_bins

        def look_at(self, stimuli: StimulusSet, number_of_trials=1) -> NeuroidAssembly:
            return self.assembly_factory(stimuli, self.time_bins)

    def test_proper_format_with_timebin(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def assembly_maker(stimuli: StimulusSet, time_bins: List[Tuple[int, int]]) -> NeuroidAssembly:
            num_units = 300
            rng = RandomState(1)
            neural_site_predictions = rng.rand(len(stimuli), num_units)
            return NeuroidAssembly([neural_site_predictions],
                                   coords={**{coord: ('presentation', stimuli[coord]) for coord in stimuli},
                                           **{'neuroid_id': ('neuroid', [f"nid{i}" for i in range(num_units)]),
                                              'region': ('neuroid', ['V1'] * num_units),
                                              'layer': ('neuroid', ['dummylayer'] * num_units),
                                              'time_bin_start': ('time_bin', [time_bins[0][0]]),
                                              'time_bin_end': ('time_bin', [time_bins[0][1]]),
                                              }},
                                   dims=['time_bin', 'presentation', 'neuroid'])

        load_mock.return_value = self.ModelMock(assembly_maker)
        generic_plugin_tests.test_look_at_neural_V1('dummy')

    def test_proper_format_without_timebin(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def assembly_maker(stimuli: StimulusSet, time_bins: List[Tuple[int, int]]) -> NeuroidAssembly:
            num_units = 300
            rng = RandomState(1)
            neural_site_predictions = rng.rand(len(stimuli), num_units)
            return NeuroidAssembly(neural_site_predictions,
                                   coords={**{coord: ('presentation', stimuli[coord]) for coord in stimuli},
                                           **{'neuroid_id': ('neuroid', [f"nid{i}" for i in range(num_units)]),
                                              'region': ('neuroid', ['V1'] * num_units),
                                              'layer': ('neuroid', ['dummylayer'] * num_units),
                                              }},
                                   dims=['presentation', 'neuroid'])

        load_mock.return_value = self.ModelMock(assembly_maker)
        generic_plugin_tests.test_look_at_neural_V1('dummy')

    def test_None_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        load_mock.return_value = self.ModelMock(lambda stimuli, time_bins: None)
        with pytest.raises(TypeError):
            generic_plugin_tests.test_look_at_neural_V1('dummy')

    def test_no_stimulus_id_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def assembly_maker(stimuli: StimulusSet, time_bins: List[Tuple[int, int]]) -> NeuroidAssembly:
            num_units = 300
            rng = RandomState(1)
            neural_site_predictions = rng.rand(len(stimuli), num_units)
            return NeuroidAssembly(neural_site_predictions,
                                   coords={**{coord: ('presentation', stimuli[coord]) for coord in stimuli
                                              if coord != 'stimulus_id'},
                                           **{'neuroid_id': ('neuroid', [f"nid{i}" for i in range(num_units)]),
                                              'region': ('neuroid', ['V1'] * num_units),
                                              'layer': ('neuroid', ['dummylayer'] * num_units),
                                              }},
                                   dims=['presentation', 'neuroid'])

        load_mock.return_value = self.ModelMock(assembly_maker)
        with pytest.raises(KeyError):
            generic_plugin_tests.test_look_at_neural_V1('dummy')

    def test_no_neuroid_id_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def assembly_maker(stimuli: StimulusSet, time_bins: List[Tuple[int, int]]) -> NeuroidAssembly:
            num_units = 300
            rng = RandomState(1)
            neural_site_predictions = rng.rand(len(stimuli), num_units)
            return NeuroidAssembly(neural_site_predictions,
                                   coords={**{coord: ('presentation', stimuli[coord]) for coord in stimuli},
                                           **{'neuroid_num': ('neuroid', [f"nnum{i}" for i in range(num_units)]),
                                              'region': ('neuroid', ['V1'] * num_units),
                                              'layer': ('neuroid', ['dummylayer'] * num_units),
                                              }},
                                   dims=['presentation', 'neuroid'])

        load_mock.return_value = self.ModelMock(assembly_maker)
        with pytest.raises(KeyError):
            generic_plugin_tests.test_look_at_neural_V1('dummy')

    def test_no_neuroid_dim_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def assembly_maker(stimuli: StimulusSet, time_bins: List[Tuple[int, int]]) -> NeuroidAssembly:
            rng = RandomState(1)
            neural_site_predictions = rng.rand(len(stimuli))
            return NeuroidAssembly(neural_site_predictions,
                                   coords={coord: ('presentation', stimuli[coord]) for coord in stimuli},
                                   dims=['presentation'])

        load_mock.return_value = self.ModelMock(assembly_maker)
        with pytest.raises(KeyError):
            generic_plugin_tests.test_look_at_neural_V1('dummy')

    def test_no_neuroid_dim_with_timebin_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def assembly_maker(stimuli: StimulusSet, time_bins: List[Tuple[int, int]]) -> NeuroidAssembly:
            rng = RandomState(1)
            neural_site_predictions = rng.rand(len(stimuli))
            return NeuroidAssembly([neural_site_predictions],
                                   coords={**{coord: ('presentation', stimuli[coord]) for coord in stimuli},
                                           **{'time_bin_start': ('time_bin', [time_bins[0][0]]),
                                              'time_bin_end': ('time_bin', [time_bins[0][1]]),
                                              }},
                                   dims=['time_bin', 'presentation'])

        load_mock.return_value = self.ModelMock(assembly_maker)
        with pytest.raises(KeyError):
            generic_plugin_tests.test_look_at_neural_V1('dummy')


@pytest.mark.travis_slow
def test_existing_model_plugin():
    command = [
        generic_plugin_tests.__file__,
        "--plugin_directory", Path(brainscore_vision.__file__).parent / 'models' / 'alexnet'
    ]
    retcode = pytest.main(command)
    assert retcode == 0, "Tests failed"  # https://docs.pytest.org/en/latest/reference/exit-codes.html
