from brainscore.assemblies.private import DicarloMajaj2015Loader, DicarloMajaj2015HighvarLoader, \
    MovshonFreemanZiemba2013Loader, ToliasCadena2017Loader, \
    DicarloMajaj2015TemporalHighvarLoader, \
    DicarloMajaj2015TemporalV4HighvarLoader, \
    DicarloMajaj2015TemporalITHighvarLoader, load_assembly
from tests.flags import private_access, memory_intense
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

    @memory_intense
    @private_access
    def test_movshonfreemanziemba2013(self):
        loader = MovshonFreemanZiemba2013Loader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'movshon.FreemanZiemba2013'
        assert set(assembly['region'].values) == {'V1', 'V2'}
        assert len(assembly['presentation']) == 450
        assert len(assembly['neuroid']) == 205

    @private_access
    def test_toliascadena2017(self):
        loader = ToliasCadena2017Loader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'tolias.Cadena2017'
        assert len(assembly['presentation']) == 6249
        assert len(assembly['neuroid']) == 166


class TestPrivate:
    @memory_intense
    @private_access
    def test_movshonfreemanziemba2013v1(self):
        assembly = load_assembly('movshon.FreemanZiemba2013.V1')
        assert set(assembly['region'].values) == {'V1'}
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 102

    @private_access
    def test_majaj2015temporal_highvar(self):
        loader = DicarloMajaj2015TemporalHighvarLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var6'
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 256
        assert len(assembly.sel(region='IT')['neuroid']) == 168
        assert len(assembly.sel(region='V4')['neuroid']) == 88

    @private_access
    def test_majaj2015temporal_v4highvar(self):
        loader = DicarloMajaj2015TemporalV4HighvarLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var6'
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 88

    @private_access
    def test_majaj2015temporal_ithighvar(self):
        loader = DicarloMajaj2015TemporalITHighvarLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var6'
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 168
