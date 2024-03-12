from collections import namedtuple

import pytest

# Note that we cannot import `from brainscore_vision.model_helpers.generic_plugin_tests import test_*` directly
# since this would expose the `test_*` methods during pytest test collection
from brainscore_vision.benchmark_helpers import generic_plugin_tests


class TestIdentifier:
    BenchmarkClass = namedtuple("DummyBenchmark", field_names=['identifier'])

    def test_no_identifier_fails(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkClass(identifier=None)
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_identifier(identifier='dummy')

    def test_proper_identifier(self, mocker):
        load_mock = mocker.patch('brainscore_vision.benchmark_helpers.generic_plugin_tests.load_benchmark')
        load_mock.return_value = self.BenchmarkClass(identifier='dummy')
        generic_plugin_tests.test_identifier('dummy')
