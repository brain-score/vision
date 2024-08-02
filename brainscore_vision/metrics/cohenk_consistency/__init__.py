from brainscore_vision import metric_registry
from .metric import ModelHumanCohenK

metric_registry['cohenk_consistency'] = ModelHumanCohenK
