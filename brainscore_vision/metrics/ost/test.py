from brainscore_vision import load_metric
from brainscore_vision.metrics.ost.metric import OSTCorrelation


def test_registered():
    assert isinstance(load_metric('ost'), OSTCorrelation)
