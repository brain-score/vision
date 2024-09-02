from brainscore_vision import benchmark_registry
from .benchmark import SSV2ActivityRecognitionAccuracy

benchmark_registry['ssv2-accuracy'] = SSV2ActivityRecognitionAccuracy