import pytest
from brainscore_vision import load_dataset


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('dicarlo.Rust2012.single', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Rust2012.array', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None


class TestRustSingle:
    @pytest.mark.private_access
    def test_dims(self):
        assembly = load_dataset('dicarlo.Rust2012.single')
        # (neuroid: 285, presentation: 1500, time_bin: 1)
        assert assembly.dims == ("neuroid", "presentation", "time_bin")
        assert len(assembly['neuroid']) == 285
        assert len(assembly['presentation']) == 1500
        assert len(assembly['time_bin']) == 1

    @pytest.mark.private_access
    def test_coords(self):
        assembly = load_dataset('dicarlo.Rust2012.single')
        assert len(set(assembly['stimulus_id'].values)) == 300
        assert len(set(assembly['neuroid_id'].values)) == 285
        assert len(set(assembly['region'].values)) == 2


class TestRustArray:
    @pytest.mark.private_access
    def test_dims(self):
        assembly = load_dataset('dicarlo.Rust2012.array')
        # (neuroid: 296, presentation: 53700, time_bin: 6)
        assert assembly.dims == ("neuroid", "presentation", "time_bin")
        assert len(assembly['neuroid']) == 296
        assert len(assembly['presentation']) == 53700
        assert len(assembly['time_bin']) == 6

    @pytest.mark.private_access
    def test_coords(self):
        assembly = load_dataset('dicarlo.Rust2012.array')
        assert len(set(assembly['stimulus_id'].values)) == 300
        assert len(set(assembly['neuroid_id'].values)) == 296
        assert len(set(assembly['animal'].values)) == 2
        assert len(set(assembly['region'].values)) == 2