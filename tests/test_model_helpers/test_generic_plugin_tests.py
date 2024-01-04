from collections import namedtuple

import pytest

from brainscore_vision import BrainModel
# Note that we cannot import `from brainscore_vision.model_helpers.generic_plugin_tests import test_*` directly
# since this would expose the `test_*` methods during pytest test collection
from brainscore_vision.model_helpers import generic_plugin_tests


class TestHasIdentifier:
    ModelClass = namedtuple("DummyModel", field_names=['identifier'])

    def test_no_identifier(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(identifier=None)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_has_identifier(identifier='dummy')

    def test_proper_identifier(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(identifier='dummy')
        generic_plugin_tests.test_has_identifier('dummy')


class TestHasVisualDegrees:
    ModelClass = namedtuple("DummyModel", field_names=['visual_degrees'])

    def test_proper_degrees(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(visual_degrees=lambda: 8)
        generic_plugin_tests.test_has_visual_degrees('dummy')

    def test_degrees_0(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(visual_degrees=lambda: 0)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_has_visual_degrees(identifier='dummy')

    def test_degrees_None(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(visual_degrees=lambda: None)
        with pytest.raises(TypeError):
            generic_plugin_tests.test_has_visual_degrees(identifier='dummy')

    def test_no_function(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = object()
        with pytest.raises(AttributeError):
            generic_plugin_tests.test_has_visual_degrees(identifier='dummy')


class TestSupportsBehaviorOrNeural:
    ModelClass = namedtuple("DummyModel", field_names=['start_task', 'start_recording'], defaults=[None, None])

    def test_supports_all(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(start_task=lambda task, fitting_stimuli=None: None,
                                                 start_recording=lambda recording_target, time_bins: None)
        generic_plugin_tests.test_supports_behavior_or_neural('dummy')

    def test_supports_behavior_only(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(start_task=lambda task, fitting_stimuli=None: None)
        generic_plugin_tests.test_supports_behavior_or_neural('dummy')

    def test_supports_behavior_label_only(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def start_task(task: BrainModel.Task, fitting_stimuli):
            if task != BrainModel.Task.label:
                raise NotImplementedError()

        load_mock.return_value = self.ModelClass(start_task=start_task)
        generic_plugin_tests.test_supports_behavior_or_neural('dummy')

    def test_supports_neural_only(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass(start_recording=lambda recording_target, time_bins: None)
        generic_plugin_tests.test_supports_behavior_or_neural('dummy')

    def test_supports_neural_V1_only(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def start_recording(recording_target: BrainModel.RecordingTarget, time_bins):
            if recording_target != BrainModel.RecordingTarget.V1:
                raise NotImplementedError()

        load_mock.return_value = self.ModelClass(start_recording=start_recording)
        generic_plugin_tests.test_supports_behavior_or_neural('dummy')


    def test_supports_neural_IT_only(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')

        def start_recording(recording_target: BrainModel.RecordingTarget, time_bins):
            if recording_target != BrainModel.RecordingTarget.IT:
                raise NotImplementedError()

        load_mock.return_value = self.ModelClass(start_recording=start_recording)
        generic_plugin_tests.test_supports_behavior_or_neural('dummy')

    def test_supports_None(self, mocker):
        load_mock = mocker.patch('brainscore_vision.model_helpers.generic_plugin_tests.load_model')
        load_mock.return_value = self.ModelClass()
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_supports_behavior_or_neural('dummy')
