from brainscore_vision import benchmark_registry
from .benchmark import Imagenet2012

benchmark_registry['fei-fei.Deng2009-top1'] = Imagenet2012