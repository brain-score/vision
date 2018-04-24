#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mkgu import benchmarks


class TestAnatomyFelleman:
    def test_equal(self):
        benchmark = benchmarks.load(data_name='Felleman1991', metric_name='edge_ratio')
        assert 1 == benchmark(benchmark._target_assembly)
