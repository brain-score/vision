from brainscore_vision import benchmark_registry
from .benchmark import Imagenet2012

benchmark_registry['ImageNet-top1'] = Imagenet2012