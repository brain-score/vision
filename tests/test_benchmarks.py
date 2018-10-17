#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pytest import approx

from brainscore import benchmarks
from brainscore.assemblies import DataAssembly, NeuroidAssembly, walk_coords
from brainscore.benchmarks import DicarloMajaj2015EarlyLateLoader, DicarloMajaj2015Loader, ToliasCadena2017Loader, \
    Benchmark
from brainscore.metrics.ceiling import NoCeiling
from brainscore.metrics.rdm import RDMCrossValidated


class TestBrainScore:
    def test_self(self):
        benchmark = benchmarks.load('brain-score')
        source = benchmarks.load_assembly('dicarlo.Majaj2015')
        source = type(source)(source.values,
                              coords={coord.replace('region', 'adjacent_coord'): (dims, values)
                                      for coord, dims, values in walk_coords(source)},
                              dims=source.dims, name=source.name)
        score = benchmark(source, transformation_kwargs=dict(
            cartesian_product_kwargs=dict(dividing_coord_names_source=['adjacent_coord'])))
        assert score == approx(1, abs=.05)


class TestMajaj2015IT:
    def test_loader(self):
        assembly = benchmarks.load_assembly('dicarlo.Majaj2015.IT')
        np.testing.assert_array_equal(assembly.dims, ['presentation', 'neuroid'])
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 256
        assert len(assembly.sel(region='IT')['neuroid']) == 168
        assert len(assembly.sel(region='V4')['neuroid']) == 88

    def test_ceiling(self):
        benchmark = benchmarks.load('dicarlo.Majaj2015.IT')
        ceiling = benchmark.ceiling
        assert ceiling.aggregation.sel(region='IT', aggregation='center') == approx(.817, abs=0.05)

    def test_self(self):
        benchmark = benchmarks.load('dicarlo.Majaj2015.IT')
        source = benchmarks.load_assembly('dicarlo.Majaj2015.IT').sel(region='IT')
        score, unceiled_score = benchmark(source, return_ceiled=True)
        assert all(score.aggregation.sel(aggregation='error') == unceiled_score.aggregation.sel(aggregation='error'))
        # ceiling should use the same rng, but different repetitions. results should overall be close to 1
        np.testing.assert_array_almost_equal(score.aggregation.sel(aggregation='center'), [1., 1.], decimal=2)
        target_array = DataAssembly(np.ones((2, 10, 256)), coords=score.values.coords, dims=score.values.dims)
        target_array.loc[dict(region='IT', neuroid_id=source.sel(region='V4')['neuroid_id'])] = np.nan
        target_array.loc[dict(region='V4', neuroid_id=source.sel(region='IT')['neuroid_id'])] = np.nan
        # .8 is not that satisfying. it seems that different repetitions lead to quite different outcomes
        assert np.isclose(score.values, target_array, atol=0.1).sum() / target_array.sum() > .8


class TestAnatomyFelleman:
    def test_equal(self):
        benchmark = benchmarks.load('Felleman1991')
        assert 1 == benchmark(benchmark._target_assembly)


class TestAssemblyLoaders:
    def test_majaj2015(self):
        loader = DicarloMajaj2015Loader()
        assembly = loader()
        assert isinstance(assembly, NeuroidAssembly)
        assert {'presentation', 'neuroid'} == set(assembly.dims)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm'

    def test_majaj2015early_late(self):
        loader = DicarloMajaj2015EarlyLateLoader()
        assembly = loader()
        assert isinstance(assembly, NeuroidAssembly)
        assert {'presentation', 'neuroid', 'time_bin'} == set(assembly.dims)
        assert len(assembly['time_bin']) == 2
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm'

    def test_toliascadena2017(self):
        loader = ToliasCadena2017Loader()
        assembly = loader()
        assert isinstance(assembly, NeuroidAssembly)
        assert {'presentation', 'neuroid'} == set(assembly.dims)
        assert not np.isnan(assembly).any()
        assert assembly.attrs['stimulus_set_name'] == 'tolias.Cadena2017'


class TestCadena2017:
    def test_loader(self):
        assembly = benchmarks.load_assembly('tolias.Cadena2017')
        np.testing.assert_array_equal(assembly.dims, ['presentation', 'neuroid'])
        assert hasattr(assembly, 'image_id')
        assert len(assembly['presentation']) == 7249
        assert len(assembly['neuroid']) == 166

    def test_ceiling(self):
        benchmark = benchmarks.load('tolias.Cadena2017')
        ceiling = benchmark.ceiling
        assert ceiling.aggregation.sel(region='IT', aggregation='center') == approx(.817, abs=0.05)

    def test_self(self):
        benchmark = benchmarks.load('tolias.Cadena2017')
        source = benchmarks.load_assembly('tolias.Cadena2017')
        score, unceiled_score = benchmark(source, return_ceiled=True)
        assert score.aggregation.sel(aggregation='error') == unceiled_score.aggregation.sel(aggregation='error')
        # ceiling should use the same rng, but different repetitions. results should overall be close to 1
        np.testing.assert_almost_equal(score.aggregation.sel(aggregation='center'), 1., decimal=1)
        target_array = DataAssembly(np.ones((10, 166)), coords=score.values.coords, dims=score.values.dims)
        # .4 is not satisfying at all. it seems that different repetitions lead to quite different outcomes
        # and especially in this dataset, the lack of repetitions might be quite crucial.
        assert np.isclose(score.values, target_array, atol=0.1).sum() / target_array.sum() > .4


class TestConstruct:
    def test_rdm(self):
        assembly = benchmarks.load_assembly('dicarlo.Majaj2015').sel(region='V4')
        metric = RDMCrossValidated()
        ceiling = NoCeiling()
        benchmark = Benchmark(name='test', target_assembly=assembly, metric=metric, ceiling=ceiling)
        score = benchmark(assembly)
        assert score.aggregation.sel(aggregation='center') == approx(1)
