from brainscore_vision import load_metric
from brainscore_vision.metrics.regression_correlation.metric import CrossRegressedCorrelation


def test_registered():
    # building the metric does not need tensorflow; fitting it does
    assert isinstance(load_metric('mask_regression'), CrossRegressedCorrelation)
