import numpy as np
import pytest

from brainscore.assemblies.private import DicarloMajaj2015Loader, DicarloMajaj2015HighvarLoader, \
    MovshonFreemanZiemba2013Loader, ToliasCadena2017Loader, \
    DicarloMajaj2015TemporalHighvarLoader, DicarloMajaj2015TemporalV4HighvarLoader, \
    DicarloMajaj2015TemporalITHighvarLoader, load_assembly
from tests.test_assemblies import check_standard_format


class TestBaseLoaders:
    def test_majaj2015(self):
        loader = DicarloMajaj2015Loader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm'
        assert len(assembly['presentation']) == 5760
        assert len(assembly['neuroid']) == 256
        assert len(assembly.sel(region='IT')['neuroid']) == 168
        assert len(assembly.sel(region='V4')['neuroid']) == 88

    def test_majaj2015_highvar(self):
        loader = DicarloMajaj2015HighvarLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var6'
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 256
        assert len(assembly.sel(region='IT')['neuroid']) == 168
        assert len(assembly.sel(region='V4')['neuroid']) == 88

    @pytest.mark.memory_intense
    @pytest.mark.private_access
    def test_movshonfreemanziemba2013(self):
        loader = MovshonFreemanZiemba2013Loader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'movshon.FreemanZiemba2013'
        assert set(assembly['region'].values) == {'V1', 'V2'}
        assert len(assembly['presentation']) == 450
        assert len(assembly['neuroid']) == 205


@pytest.mark.private_access
class TestToliasCadena2017:
    def test(self):
        loader = ToliasCadena2017Loader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'tolias.Cadena2017'
        assert len(assembly['presentation']) == 6249
        assert len(assembly['neuroid']) == 166


@pytest.mark.private_access
@pytest.mark.memory_intense
class TestMajaj2015:
    def test_majaj2015TemporalHighvar(self):
        loader = DicarloMajaj2015TemporalHighvarLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var6'
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 256
        assert len(assembly.sel(region='IT')['neuroid']) == 168
        assert len(assembly.sel(region='V4')['neuroid']) == 88

    def test_majaj2015V4TemporalHighvar(self):
        loader = DicarloMajaj2015TemporalV4HighvarLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var6'
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 88
        assert len(assembly['time_bin']) == 39
        np.testing.assert_array_equal(assembly['time_bin_start'], list(range(-100, 281, 10)))
        np.testing.assert_array_equal(assembly['time_bin_end'], list(range(-80, 301, 10)))

    def test_majaj2015ITTemporalHighvar(self):
        loader = DicarloMajaj2015TemporalITHighvarLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var6'
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 168
        assert len(assembly['time_bin']) == 39
        np.testing.assert_array_equal(assembly['time_bin_start'], list(range(-100, 281, 10)))
        np.testing.assert_array_equal(assembly['time_bin_end'], list(range(-80, 301, 10)))


@pytest.mark.private_access
@pytest.mark.memory_intense
class TestFreemanZiemba2013:
    def test_v1(self):
        assembly = load_assembly('movshon.FreemanZiemba2013.private.V1')
        assert set(assembly['region'].values) == {'V1'}
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 102

    def test_temporal_v1(self):
        assembly = load_assembly('movshon.FreemanZiemba2013.temporal.private.V1')
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 102
        np.testing.assert_array_equal(assembly['time_bin_start'], list(range(0, 291, 10)))
        np.testing.assert_array_equal(assembly['time_bin_end'], list(range(10, 301, 10)))

    def test_temporal_v2(self):
        assembly = load_assembly('movshon.FreemanZiemba2013.temporal.private.V2')
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 103
        np.testing.assert_array_equal(assembly['time_bin_start'], list(range(0, 291, 10)))
        np.testing.assert_array_equal(assembly['time_bin_end'], list(range(10, 301, 10)))
