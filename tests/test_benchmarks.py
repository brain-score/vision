#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pytest import approx

from brainscore import benchmarks
from brainscore.assemblies import DataAssembly, walk_coords
from brainscore.benchmarks import SplitBenchmark, metrics
from brainscore.metrics.ceiling import SplitNoCeiling


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


class TestMajaj2015:
    def test_loader(self):
        assembly = benchmarks.load_assembly('dicarlo.Majaj2015')
        np.testing.assert_array_equal(assembly.dims, ['presentation', 'neuroid'])
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 256
        assert len(assembly.sel(region='IT')['neuroid']) == 168
        assert len(assembly.sel(region='V4')['neuroid']) == 88

    def test_ceiling(self):
        benchmark = benchmarks.load('dicarlo.Majaj2015')
        ceiling = benchmark.ceiling
        assert ceiling.aggregation.sel(region='IT', aggregation='center') == approx(.817, abs=0.05)

    def test_self(self):
        benchmark = benchmarks.load('dicarlo.Majaj2015')
        source = benchmarks.load_assembly('dicarlo.Majaj2015')
        score, unceiled_score = benchmark(source, return_ceiled=True)
        assert all(score.aggregation.sel(aggregation='error') == unceiled_score.aggregation.sel(aggregation='error'))
        # ceiling should use the same rng, but different repetitions. results should overall be close to 1
        np.testing.assert_array_almost_equal(score.aggregation.sel(aggregation='center'), [1., 1.], decimal=2)
        target_array = DataAssembly(np.ones((2, 10, 256)), coords=score.values.coords, dims=score.values.dims)
        target_array.loc[dict(region='IT', neuroid_id=source.sel(region='V4')['neuroid_id'])] = np.nan
        target_array.loc[dict(region='V4', neuroid_id=source.sel(region='IT')['neuroid_id'])] = np.nan
        # .8 is not that satisfying. it seems that different repetitions lead to quite different outcomes
        assert np.isclose(score.values, target_array, atol=0.1).sum() / target_array.sum() > .8

    def test_construct_kwargs(self):
        assembly = benchmarks.load_assembly('dicarlo.Majaj2015')
        assembly = assembly.rename({'presentation': 'stimulus'})
        metric = metrics['rdm']()
        dimensions = ('stimulus', 'neuroid')
        benchmark = SplitBenchmark(target_assembly=assembly, metric=metric, ceiling=SplitNoCeiling(),
                                   target_splits=['region'], target_splits_kwargs=dict(non_dividing_dims=dimensions))
        score = benchmark(assembly, transformation_kwargs=dict(
            alignment_kwargs=dict(order_dimensions=dimensions),
            cartesian_product_kwargs=dict(non_dividing_dims=dimensions)))
        np.testing.assert_array_equal(score.aggregation.sel(aggregation='center').dims, ['region'])
        np.testing.assert_array_equal(score.values.dims, ['region', 'split'])


class TestAnatomyFelleman:
    def test_equal(self):
        benchmark = benchmarks.load('Felleman1991')
        assert 1 == benchmark(benchmark._target_assembly)


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
        score, unceiled_score = benchmark(source, return_unceiled=True)
        assert score.aggregation.sel(aggregation='error') == unceiled_score.aggregation.sel(aggregation='error')
        # ceiling should use the same rng, but different repetitions. results should overall be close to 1
        np.testing.assert_almost_equal(score.aggregation.sel(aggregation='center'), 1., decimal=1)
        target_array = DataAssembly(np.ones((10, 166)), coords=score.values.coords, dims=score.values.dims)
        # .4 is not satisfying at all. it seems that different repetitions lead to quite different outcomes
        # and especially in this dataset, the lack of repetitions might be quite crucial.
        assert np.isclose(score.values, target_array, atol=0.1).sum() / target_array.sum() > .4
