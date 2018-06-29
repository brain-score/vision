#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
from pytest import approx

from brainscore import benchmarks
from brainscore.assemblies import NeuroidAssembly
from brainscore.benchmarks import SplitBenchmark, metrics
from brainscore.metrics.ceiling import SplitNoCeiling


def test_score_precomputed_alexnet_activations():
    # load activations
    activations = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'alexnet-pca200.pkl'))
    # score
    benchmark = benchmarks.load('dicarlo.Majaj2015', 'neural_fit')
    scores = None
    for layer in np.unique(activations['layers']):
        layer_activations = activations[activations['layers'] == layer]
        layer_scores = benchmark(layer_activations)
        layer_scores['layer'] = [layer] * len(layer_scores)
        if scores is None:
            scores = layer_scores
        else:
            scores.append(layer_scores)
    # eval
    region_best_layers = {'V4': 'conv1', 'IT': 'conv5'}
    for region in scores.region:
        max_region_score = scores[scores['region'] == region].max()
        assert max_region_score['layer'] == region_best_layers[region]


class TestRDMBenchmark(object):
    class TestBenchmark(SplitBenchmark):
        def __init__(self, assembly):
            metric = metrics['rdm']()
            ceiling = SplitNoCeiling()
            super().__init__(assembly, metric, ceiling, target_splits=())

    def _build_benchmark(self, assembly):
        return self.TestBenchmark(assembly)

    def _score(self, source_assembly, target_assembly):
        benchmark = self._build_benchmark(target_assembly)
        return benchmark(source_assembly)

    def test_2d_equal(self):
        values = np.random.rand(60, 30)
        assembly = NeuroidAssembly(values, coords={
            'image_id': ('presentation', list(range(60))), 'object_name': ('presentation', ['A', 'B'] * 30),
            'neuroid_id': ('neuroid', list(range(30)))},
                                   dims=['presentation', 'neuroid'])
        score = self._score(assembly, assembly)
        assert score.aggregation.sel(aggregation='center') == approx(1.)

    def test_3d_equal(self):
        # presentation x neuroid
        values = np.broadcast_to(np.random.rand(60, 30, 1), [60, 30, 3]).copy()
        # can't stack because xarray gets confused with repeated dimensions
        assembly1 = NeuroidAssembly(values, coords={
            'image_id': ('presentation', list(range(60))), 'object_name': ('presentation', ['A', 'B'] * 30),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim1': list(range(3))},
                                    dims=['presentation', 'neuroid', 'dim1'])
        assembly2 = NeuroidAssembly(values, coords={
            'image_id': ('presentation', list(range(60))), 'object_name': ('presentation', ['A', 'B'] * 30),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim2': list(range(3))},
                                    dims=['presentation', 'neuroid', 'dim2'])
        score = self._score(assembly1, assembly2)
        score = score.aggregation.sel(aggregation='center')
        np.testing.assert_array_equal(score.shape, [3, 3])
        np.testing.assert_array_almost_equal(score, np.broadcast_to(1, [3, 3]))

    def test_3d_diag(self):
        # presentation x neuroid
        values = np.random.rand(60, 30, 3)
        assembly1 = NeuroidAssembly(values, coords={
            'image_id': ('presentation', list(range(60))), 'object_name': ('presentation', ['A', 'B'] * 30),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim1': list(range(3))},
                                    dims=['presentation', 'neuroid', 'dim1'])
        assembly2 = NeuroidAssembly(values, coords={
            'image_id': ('presentation', list(range(60))), 'object_name': ('presentation', ['A', 'B'] * 30),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim2': list(range(3))},
                                    dims=['presentation', 'neuroid', 'dim2'])
        score = self._score(assembly1, assembly2)
        center = score.aggregation.sel(aggregation='center')
        np.testing.assert_array_equal(center.shape, [3, 3])
        assert len(score.values['split']) * 3 == (score.values == approx(1)).sum()
        diags_indexer = center['dim1'] == center['dim2']
        diag_values = center.values[diags_indexer.values]
        np.testing.assert_array_almost_equal(diag_values, 1)

    def test_3d_equal_presentation_last(self):
        values = np.broadcast_to(np.random.rand(60, 30, 1), [60, 30, 3]).copy()
        assembly1 = NeuroidAssembly(values.T, coords={
            'image_id': ('presentation', list(range(60))), 'object_name': ('presentation', ['A', 'B'] * 30),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim1': list(range(3))},
                                    dims=['dim1', 'neuroid', 'presentation'])
        assembly2 = NeuroidAssembly(values.T, coords={
            'image_id': ('presentation', list(range(60))), 'object_name': ('presentation', ['A', 'B'] * 30),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim2': list(range(3))},
                                    dims=['dim2', 'neuroid', 'presentation'])
        scores = self._score(assembly1, assembly2)
        scores = scores.aggregation.sel(aggregation='center')
        np.testing.assert_array_equal(scores.shape, [3, 3])
        np.testing.assert_array_almost_equal(scores, 1)
