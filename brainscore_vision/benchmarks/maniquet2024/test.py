#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:22:59 2024

@author: costantino_ai
"""

import pytest
from brainscore_vision import benchmark_registry

@pytest.mark.parametrize('benchmark', [
    'Maniquet2024ConfusionSimilarity',
    'Maniquet2024TasksConsistency',
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry
