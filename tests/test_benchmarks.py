#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pytest import approx

from brainscore import benchmarks


class TestAnatomyFelleman:
    def test_equal(self):
        benchmark = benchmarks.load('Felleman1991')
        assert 1 == benchmark(benchmark._target_assembly)


class TestMajaj2015:
    def test_target_assembly(self):
        benchmark = benchmarks.load('dicarlo.Majaj2015-rdm')
        np.testing.assert_array_equal(benchmark._target_assembly.dims, ['presentation', 'neuroid'])
        assert len(benchmark._target_assembly['presentation']) == 2560
        assert len(benchmark._target_assembly['neuroid']) == 296
        assert len(benchmark._target_assembly.sel(region='IT')['neuroid']) == 168
        assert len(benchmark._target_assembly.sel(region='V4')['neuroid']) == 128

    def test_ceiling(self):
        benchmark = benchmarks.load('dicarlo.Majaj2015')
        ceiling = benchmark.ceiling
        assert ceiling.aggregation.sel(region='IT', aggregation='center') == approx(.817, abs=0.05)

    def test_self(self):
        benchmark = benchmarks.load('dicarlo.Majaj2015')
        source = benchmark.average_repetition(benchmark._target_assembly)
        score, unceiled_score = benchmark(source, return_unceiled=True)
        assert all(score.aggregation.sel(aggregation='error') == unceiled_score.aggregation.sel(aggregation='error'))
        np.testing.assert_array_almost_equal(score.aggregation.sel(aggregation='center'), [1., 1.], decimal=2)
