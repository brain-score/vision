#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from mkgu import benchmarks


class TestAnatomyFelleman:
    def test_equal(self):
        benchmark = benchmarks.load(data_name='Felleman1991', metric_name='edge_ratio')
        assert 1 == benchmark(benchmark._target_assembly)


class TestMajaj2015:
    def test_target_assembly(self):
        benchmark = benchmarks.load(data_name='dicarlo.Majaj2015', metric_name='rdm')
        np.testing.assert_array_equal(benchmark._target_assembly.dims, ['presentation', 'neuroid'])
        assert len(benchmark._target_assembly['presentation']) == 2560
        assert len(benchmark._target_assembly['neuroid']) == 296
        assert len(benchmark._target_assembly.sel(region='IT')['neuroid']) == 168
        assert len(benchmark._target_assembly.sel(region='V4')['neuroid']) == 128
