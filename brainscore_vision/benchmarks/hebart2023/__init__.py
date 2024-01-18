from brainscore_vision import benchmark_registry
from . import benchmark

# TODO - one for dot and one for cosine?
benchmark_registry[''] = getattr(benchmark, "")
benchmark_registry[''] = getattr(benchmark, "")
