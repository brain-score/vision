from brainscore.assemblies.public import load_assembly, \
    DicarloMajaj2015TemporalLowvarLoader, DicarloMajaj2015TemporalV4LowvarLoader, DicarloMajaj2015TemporalITLowvarLoader
from tests.flags import private_access
from tests.test_assemblies import check_standard_format


def test_Majaj2015ITLowvar():
    assembly = load_assembly('dicarlo.Majaj2015.lowvar.IT')
    assert set(assembly['region'].values) == {'IT'}
    assert len(assembly['presentation']) == 3200
    assert len(assembly['neuroid']) == 168


class TestMajaj2015Temporal:
    @private_access
    def test_majaj2015temporal_lowvar(self):
        loader = DicarloMajaj2015TemporalLowvarLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var03'
        assert len(assembly['presentation']) == 3200
        assert len(assembly['neuroid']) == 256
        assert len(assembly.sel(region='IT')['neuroid']) == 168
        assert len(assembly.sel(region='V4')['neuroid']) == 88

    @private_access
    def test_majaj2015temporal_v4lowvar(self):
        loader = DicarloMajaj2015TemporalV4LowvarLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var03'
        assert len(assembly['presentation']) == 3200
        assert len(assembly['neuroid']) == 88

    @private_access
    def test_majaj2015temporal_itlowvar(self):
        loader = DicarloMajaj2015TemporalITLowvarLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var03'
        assert len(assembly['presentation']) == 3200
        assert len(assembly['neuroid']) == 168
