from brainscore_vision import metric_registry
from .metric import BakerAccuracyDelta

metric_registry['baker_accuracy_delta'] = BakerAccuracyDelta
