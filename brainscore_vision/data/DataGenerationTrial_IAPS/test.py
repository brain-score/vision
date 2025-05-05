import pytest

from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers import check_standard_format


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_assembly():
    assembly = load_dataset('DataGenerationTrial_IAPS')
    check_standard_format(assembly, nans_expected=True)
    assert assembly.attrs['stimulus_set_identifier'] == 'DataGenerationTrial_IAPS'
    assert set(assembly['region'].values) == {'IT'}
    assert len(assembly['presentation']) == 30150
    assert len(assembly['neuroid']) == 36


@pytest.mark.private_access
def test_stimulus_set():
    stimulus_set = load_stimulus_set('DataGenerationTrial_IAPS')
    assert len(stimulus_set) == 10

        