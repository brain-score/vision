import pytest

from brainscore_vision.benchmarks.cadena2017.benchmark import AssemblyLoader
from . import check_standard_format


@pytest.mark.private_access
class TestAssembly:
    def test(self):
        loader = AssemblyLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_identifier'] == 'tolias.Cadena2017'
        assert len(assembly['presentation']) == 6249
        assert len(assembly['neuroid']) == 166
