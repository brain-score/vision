
import pytest

from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers import check_standard_format


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_assembly():
    assembly = load_dataset('IAPS')
    check_standard_format(assembly, nans_expected=True)
    assert assembly.attrs['stimulus_set_identifier'] == 'IAPS'
    assert set(assembly['region'].values) == {'IT'}
    assert len(assembly['presentation']) == 30150
    assert len(assembly['neuroid']) == 36
    # assert len(set(assembly['background_id'].values)) == 121


@pytest.mark.private_access
def test_stimulus_set():
    stimulus_set = load_stimulus_set('IAPS')
    assert len(stimulus_set) == 10

        