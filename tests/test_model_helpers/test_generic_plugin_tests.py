from collections import namedtuple

import pytest

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
