import pytest

from brainscore.benchmarks.freemanziemba2013 import load_assembly
from . import check_standard_format


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestAssembly:
    def test_V1(self):
        assembly = load_assembly(region='V1', average_repetitions=True)
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'movshon.FreemanZiemba2013.aperture-private'
        assert set(assembly['region'].values) == {'V1'}
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 102

    def test_V2(self):
        assembly = load_assembly(region='V2', average_repetitions=True)
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'movshon.FreemanZiemba2013.aperture-private'
        assert set(assembly['region'].values) == {'V2'}
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 103
