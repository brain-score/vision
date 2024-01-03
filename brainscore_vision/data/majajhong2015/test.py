import pytest

from brainscore_vision.benchmarks.majajhong2015.benchmark import load_assembly
from brainscore_vision import load_dataset
from brainscore_vision.benchmark_helpers import check_standard_format


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


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('MajajHong2015', marks=[pytest.mark.private_access]),
    pytest.param('MajajHong2015.public', marks=[]),
    pytest.param('MajajHong2015.private', marks=[pytest.mark.private_access]),
    pytest.param('MajajHong2015.temporal', marks=[pytest.mark.private_access, pytest.mark.memory_intense]),
    pytest.param('MajajHong2015.temporal.public', marks=[pytest.mark.memory_intense]),
    pytest.param('MajajHong2015.temporal.private',
                 marks=[pytest.mark.private_access, pytest.mark.memory_intense]),
    pytest.param('MajajHong2015.temporal-10ms', marks=[pytest.mark.private_access, pytest.mark.memory_intense]),
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None
