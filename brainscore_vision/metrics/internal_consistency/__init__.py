from brainscore_vision import metric_registry
from .ceiling import InternalConsistency

metric_registry['internal_consistency'] = InternalConsistency
