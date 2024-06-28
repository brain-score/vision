import numpy as np
import pytest

from brainscore_vision import load_dataset


@pytest.mark.private_access
def test_existence():
    assert load_dataset('Seibert2019') is not None


@pytest.mark.private_access
class TestSeibert:
    def test_dims(self):
        assembly = load_dataset('Seibert2019')
        # neuroid: 258 presentation: 286080 time_bin: 1
        assert assembly.dims == ("neuroid", "presentation", "time_bin")
        assert len(assembly['neuroid']) == 258
        assert len(assembly['presentation']) == 286080
        assert len(assembly['time_bin']) == 1

    def test_coords(self):
        assembly = load_dataset('Seibert2019')
        assert len(set(assembly['stimulus_id'].values)) == 5760
        assert len(set(assembly['neuroid_id'].values)) == 258
        assert len(set(assembly['animal'].values)) == 3
        assert len(set(assembly['region'].values)) == 2
        assert len(set(assembly['variation'].values)) == 3

    def test_content(self):
        assembly = load_dataset('Seibert2019')
        assert np.count_nonzero(np.isnan(assembly)) == 19118720
        assert assembly.stimulus_set_identifier == "hvm"
        hvm = assembly.stimulus_set
        assert hvm.shape == (5760, 19)
