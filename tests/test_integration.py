#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import xarray as xr

from mkgu.metrics import benchmarks


def test_score_precomputed_alexnet_activations():
    # load activations
    activations = xr.open_dataarray(os.path.join(os.path.dirname(__file__), 'alexnet-pca200.nc'))
    activations = activations[activations['var'] == 'V6']
    # score
    benchmark = benchmarks.load('dicarlo/hong2014', 'neural_fit')
    scores = None
    for layer in np.unique(activations['layer']):
        layer_activations = activations.T[activations['layer'] == layer].T
        layer_scores = benchmark(layer_activations)
        layer_scores['layer'] = layer
        scores = xr.concat([scores, layer_scores], dim='layer') if scores is not None else layer_scores
    # evaluate
    expected = xr.DataArray([0.54, 0.58],
                            coords={'region': ['V4', 'IT'],
                                    'layer': ('region', ['conv1', 'conv5'])},
                            dims=['region'])
    expected_best_layers = {'V4': 'conv1', 'IT': 'conv5'}
    actual_best_layers = {}
    for region in expected_best_layers:
        max_region_score = scores.sel(region=region).max()
        layer = scores.where(scores == max_region_score, drop=True)['layer']
        actual_best_layers[region] = layer.values[0]
    assert expected_best_layers == actual_best_layers, "{} != {}".format(expected_best_layers, actual_best_layers)
