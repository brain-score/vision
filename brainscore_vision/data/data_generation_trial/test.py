
import pytest

from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers import check_standard_format


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_assembly():
    assembly = load_dataset('Co3D')
    check_standard_format(assembly, nans_expected=True)
    assert assembly.attrs['stimulus_set_identifier'] == 'Co3D'
    assert set(assembly['region'].values) == {'IT'}
    assert len(assembly['presentation']) == 7588
    assert len(assembly['neuroid']) == 36
    # assert len(set(assembly['background_id'].values)) == 121


@pytest.mark.private_access
def test_stimulus_set():
    stimulus_set = load_stimulus_set('Co3D')
    assert len(stimulus_set) == 319

        