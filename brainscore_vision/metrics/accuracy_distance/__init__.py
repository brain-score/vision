from brainscore_vision import metric_registry
from .metric import AccuracyDistance

metric_registry['accuracy_distance'] = AccuracyDistance
