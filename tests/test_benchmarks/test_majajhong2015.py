import pytest

from brainscore.benchmarks.majajhong2015 import load_assembly
from . import check_standard_format


@pytest.mark.private_access
class TestAssembly:
    def test_majajhong2015_V4(self):
        assembly = load_assembly(average_repetitions=True, region='V4')
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_identifier'] == 'dicarlo.hvm-private'
        assert set(assembly['region'].values) == {'V4'}
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 88

    def test_majajhong2015_IT(self):
        assembly = load_assembly(average_repetitions=True, region='IT')
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_identifier'] == 'dicarlo.hvm-private'
        assert set(assembly['region'].values) == {'IT'}
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 168
