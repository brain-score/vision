from brainscore_vision import metric_registry
from .metric import Threshold, ThresholdElevation

metric_registry['threshold'] = Threshold
metric_registry['threshold_elevation'] = ThresholdElevation
