from brainscore_vision import metric_registry
from .ceiling import RSACeiling

metric_registry['rsa_ceiling'] = RSACeiling
