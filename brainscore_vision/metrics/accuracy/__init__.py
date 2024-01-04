from brainscore_vision import metric_registry
from .metric import Accuracy

metric_registry['accuracy'] = Accuracy
