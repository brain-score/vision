import numpy as np

from brainio_base.assemblies import NeuroidAssembly
from brainscore.benchmarks.loaders import DicarloMajaj2015Loader, MovshonFreemanZiemba2013Loader, ToliasCadena2017Loader
from tests import private_access


def check_standard_format(assembly):
    assert isinstance(assembly, NeuroidAssembly)
    assert set(assembly.dims).issuperset({'presentation', 'neuroid'})
    assert hasattr(assembly, 'image_id')
    assert hasattr(assembly, 'neuroid_id')
    assert not np.isnan(assembly).any()
    assert 'stimulus_set_name' in assembly.attrs


class TestAssemblyLoaders:
    def test_majaj2015(self):
        loader = DicarloMajaj2015Loader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-V6'
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 256
        assert len(assembly.sel(region='IT')['neuroid']) == 168
        assert len(assembly.sel(region='V4')['neuroid']) == 88

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
