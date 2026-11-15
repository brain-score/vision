# the numeric tests are part of /benchmarks/baker2022/test.py
from brainscore_vision import load_metric
from brainscore_vision.metrics.baker_accuracy_delta.metric import BakerAccuracyDelta


def test_registered():
    metric = load_metric('baker_accuracy_delta', image_types=['w', 'f'])
    assert isinstance(metric, BakerAccuracyDelta)
    assert metric.image_types == ['w', 'f']
