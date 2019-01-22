#!/usr/bin/env python
# -*- coding: utf-8 -*-
import brainscore
from pytest import approx

from brainscore import benchmarks
from brainscore.benchmarks import split_assembly


class TestSplitAssembly:
    def test_repeat(self):
        assembly = brainscore.get_assembly('dicarlo.Majaj2015')
        splits0 = split_assembly(assembly)
        splits1 = split_assembly(assembly)
        assert len(splits0) == len(splits1)
        assert all(s0.equals(s1) for s0, s1 in zip(splits0, splits1))


class TestMajaj2015:
    def test_ceiling_V4(self):
        benchmark = benchmarks.load('dicarlo.Majaj2015.V4')
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.892, abs=0.01)

    def test_ceiling_IT(self):
        benchmark = benchmarks.load('dicarlo.Majaj2015.IT')
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.817, abs=0.01)

    def test_self(self):
        benchmark = benchmarks.load('dicarlo.Majaj2015.IT')
        source = benchmark.assembly
        source.name = 'dicarlo.Majaj2015.IT'
        score = benchmark(source).raw
        assert score.sel(aggregation='center') == approx(1)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(0), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(0), "too much deviation between neuroids"


class TestCadena2017:
    def test_ceiling(self):
        benchmark = benchmarks.load('tolias.Cadena2017')
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.577, abs=0.05)

    def test_self(self):
        benchmark = benchmarks.load('tolias.Cadena2017')
        source = benchmark.assembly
        score = benchmark(source).raw
        assert score.sel(aggregation='center') == approx(1)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(0)
        assert raw_values.mean('split').std() == approx(0)
