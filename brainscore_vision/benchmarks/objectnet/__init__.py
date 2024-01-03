from brainscore_vision import benchmark_registry
from .benchmark import Objectnet

benchmark_registry['ObjectNet-top1'] = Objectnet