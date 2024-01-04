from collections import namedtuple

import pytest

from brainscore_vision.model_helpers import generic_plugin_tests


# Note that we cannot import `from brainscore_vision.model_helpers.generic_plugin_tests import test_*` directly
# since this would expose the `test_*` methods during pytest test collection

class TestHasIdentifier:
    ModelClass = namedtuple("DummyModel", field_names=['identifier'])

    def test_no_identifier(self, mocker):
        load_mock = mocker.patch('brainscore_vision.load_model')
        load_mock.returnvalue = self.ModelClass(identifier=None)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_has_identifier(identifier='dummy')

    def test_proper_identifier(self, mocker):
        load_mock = mocker.patch('brainscore_vision.load_model')
        load_mock.returnvalue = self.ModelClass(identifier='dummy')
        generic_plugin_tests.test_has_identifier('dummy')
