import numpy as np

from brainio_base.assemblies import NeuroidAssembly
from brainscore.benchmarks import DicarloMajaj2015Loader, ToliasCadena2017Loader


class TestAssemblyLoaders:
    def test_majaj2015(self):
        loader = DicarloMajaj2015Loader()
        assembly = loader()
        assert isinstance(assembly, NeuroidAssembly)
        assert {'presentation', 'neuroid'} == set(assembly.dims)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm'
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 256
        assert len(assembly.sel(region='IT')['neuroid']) == 168
        assert len(assembly.sel(region='V4')['neuroid']) == 88

    def test_toliascadena2017(self):
        loader = ToliasCadena2017Loader()
        assembly = loader()
        assert isinstance(assembly, NeuroidAssembly)
        assert {'presentation', 'neuroid'} == set(assembly.dims)
        assert not np.isnan(assembly).any()
        assert assembly.attrs['stimulus_set_name'] == 'tolias.Cadena2017'
        assert hasattr(assembly, 'image_id')
        assert len(assembly['presentation']) == 6249
        assert len(assembly['neuroid']) == 166
