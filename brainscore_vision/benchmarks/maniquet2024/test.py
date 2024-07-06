#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:22:59 2024

@author: costantino_ai
"""

import pytest
from brainscore_vision import load_benchmark

@pytest.mark.parametrize('benchmark', [
    'Maniquet2024-confusion_similarity',
    'Maniquet2024-tasks_consistency',
])
def test_benchmark_registry(benchmark):
    assert load_benchmark(benchmark) is not None
