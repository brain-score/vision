#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:40:53 2024

@author: costantino_ai
"""

import pytest

from brainscore_vision import load_metric
from brainscore_vision.metrics.maniquet2024_metrics.metric import ConfusionSimilarity, TasksConsistency


@pytest.mark.parametrize('identifier, metric_class', [
    ('confusion_similarity', ConfusionSimilarity),
    ('tasks_consistency', TasksConsistency),
])
def test_registered(identifier, metric_class):
    assert isinstance(load_metric(identifier), metric_class)
