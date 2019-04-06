import numpy as np

from brainscore.assemblies.public import load_assembly
from tests.flags import private_access, memory_intense


def test_Majaj2015ITLowvar():
    assembly = load_assembly('dicarlo.Majaj2015.lowvar.IT')
    assert set(assembly['region'].values) == {'IT'}
    assert len(assembly['presentation']) == 3200
    assert len(assembly['neuroid']) == 168


@private_access
@memory_intense
class TestMajaj2015Temporal:
    def test_majaj2015TemporalLowvar(self):
        assembly = load_assembly('dicarlo.Majaj2015.temporal.lowvar')
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var03'
        assert len(assembly['presentation']) == 3200
        assert len(assembly['neuroid']) == 256
        assert len(assembly.sel(region='IT')['neuroid']) == 168
        assert len(assembly.sel(region='V4')['neuroid']) == 88
        assert len(assembly['time_bin']) == 39
        np.testing.assert_array_equal(assembly['time_bin_start'], list(range(-100, 281, 10)))
        np.testing.assert_array_equal(assembly['time_bin_end'], list(range(-80, 301, 10)))

    def test_majaj2015ITTemporalLowvar(self):
        assembly = load_assembly('dicarlo.Majaj2015.temporal.lowvar.IT')
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var03'
        assert set(assembly['region'].values) == {'IT'}
        assert len(assembly['presentation']) == 3200
        assert len(assembly['neuroid']) == 168
        assert len(assembly['time_bin']) == 39
        np.testing.assert_array_equal(assembly['time_bin_start'], list(range(-100, 281, 10)))
        np.testing.assert_array_equal(assembly['time_bin_end'], list(range(-80, 301, 10)))

    def test_majaj2015V4TemporalLowvar(self):
        assembly = load_assembly('dicarlo.Majaj2015.temporal.lowvar.V4')
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var03'
        assert set(assembly['region'].values) == {'V4'}
        assert len(assembly['presentation']) == 3200
        assert len(assembly['neuroid']) == 88
        assert len(assembly['time_bin']) == 39
        np.testing.assert_array_equal(assembly['time_bin_start'], list(range(-100, 281, 10)))
        np.testing.assert_array_equal(assembly['time_bin_end'], list(range(-80, 301, 10)))
