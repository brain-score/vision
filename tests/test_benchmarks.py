#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from pytest import approx

from brainscore import benchmarks
from brainscore.assemblies import NeuroidAssembly
from brainscore.benchmarks import Benchmark, DicarloMajaj2015Loader
from brainscore.contrib.benchmarks import DicarloMajaj2015EarlyLateLoader, ToliasCadena2017Loader, \
    MovshonFreemanZiemba2013Loader
from brainscore.metrics.ceiling import NoCeiling
from brainscore.metrics.rdm import RDMCrossValidated


class TestMajaj2015:
    def test_loader(self):
        assembly = benchmarks.load_assembly('dicarlo.Majaj2015')
        np.testing.assert_array_equal(assembly.dims, ['presentation', 'neuroid'])
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 256
        assert len(assembly.sel(region='IT')['neuroid']) == 168
        assert len(assembly.sel(region='V4')['neuroid']) == 88

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
        source = benchmarks.load_assembly('dicarlo.Majaj2015').sel(region='IT')
        source.name = 'dicarlo.Majaj2015.IT'
        score = benchmark(source)
        # .8 is not that satisfying. it seems that different repetitions lead to quite different outcomes
        assert score.sel(aggregation='center') == approx(.82, abs=.01), "overall score too low"
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(0.007, abs=0.001), "too much deviation between splits"
        # .16 is actually a lot of deviation between different neuroids...
        assert raw_values.mean('split').std() == approx(0.16, abs=0.01), "too much deviation between neuroids"


@pytest.mark.skip(reason="ignore anatomy for now")
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

    def test_movshonfreemanziemba2013(self):
        loader = MovshonFreemanZiemba2013Loader()
        assembly = loader()
        assert isinstance(assembly, NeuroidAssembly)
        assert {'presentation', 'neuroid'} == set(assembly.dims)
        assert not np.isnan(assembly).any()
        assert assembly.attrs['stimulus_set_name'] == 'movshon.FreemanZiemba2013'
        assert len(assembly['presentation']) == 450
        assert len(assembly['neuroid']) == 205


class TestCadena2017:
    def test_loader(self):
        assembly = benchmarks.load_assembly('tolias.Cadena2017')
        np.testing.assert_array_equal(assembly.dims, ['presentation', 'neuroid'])
        assert hasattr(assembly, 'image_id')
        assert len(assembly['presentation']) == 6249
        assert len(assembly['neuroid']) == 166

    def test_ceiling(self):
        benchmark = benchmarks.load('tolias.Cadena2017')
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.577, abs=0.05)

    def test_self(self):
        benchmark = benchmarks.load('tolias.Cadena2017')
        source = benchmarks.load_assembly('tolias.Cadena2017')
        score = benchmark(source)
        # .8 is not that satisfying. it seems that different repetitions lead to quite different outcomes
        assert score.sel(aggregation='center') == approx(.58, abs=.01), "overall score too low"
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(0.007, abs=0.001), "too much deviation between splits"
        # .22 is a lot of deviation between different neuroids...
        assert raw_values.mean('split').std() == approx(0.21, abs=0.01), "too much deviation between neuroids"


class TestConstruct:
    def test_rdm(self):
        assembly = benchmarks.load_assembly('dicarlo.Majaj2015').sel(region='V4')
        metric = RDMCrossValidated()
        ceiling = NoCeiling()
        benchmark = Benchmark(name='test', target_assembly=assembly, metric=metric, ceiling=ceiling)
        score = benchmark(assembly)
        assert score.sel(aggregation='center') == approx(1)
