#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd

from brainscore import benchmarks


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
