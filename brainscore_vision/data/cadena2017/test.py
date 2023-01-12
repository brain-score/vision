import pytest

from brainscore_vision.benchmarks.cadena2017.benchmark import AssemblyLoader
from brainscore_vision.benchmark_helpers import check_standard_format
from brainscore_vision import load_dataset


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('tolias.Cadena2017', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None


@pytest.mark.private_access
class TestAssembly:
    def test(self):
        loader = AssemblyLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_identifier'] == 'tolias.Cadena2017'
        assert len(assembly['presentation']) == 6249
        assert len(assembly['neuroid']) == 166
