from brainscore_vision import benchmark_registry
from .benchmark import PhysionGlobalPredictionAccuracy
from .benchmark import PhysionGlobalDetectionAccuracy
from .benchmark import PhysionSnippetDetectionAccuracy

benchmark_registry['Physionv1.5-ocd'] = PhysionGlobalDetectionAccuracy
benchmark_registry['Physionv1.5-ocp'] = PhysionGlobalPredictionAccuracy
benchmark_registry['Physionv1.5-snippet-rollout-performance'] = PhysionSnippetDetectionAccuracy