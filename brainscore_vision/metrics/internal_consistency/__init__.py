from brainscore_vision import metric_registry
from .ceiling import InternalConsistency

from brainscore_vision.metric_helpers.temporal import PerTime


metric_registry['internal_consistency'] = InternalConsistency
metric_registry['internal_consistency_temporal'] = lambda *args, **kwargs: PerTime(InternalConsistency(*args, **kwargs))