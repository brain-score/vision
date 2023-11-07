from brainscore_vision import benchmark_registry
from .benchmarks.benchmark import MovshonFreemanZiemba2013V1PLS, MovshonFreemanZiemba2013V2PLS
from .benchmarks.public_benchmarks import FreemanZiembaV1PublicBenchmark, FreemanZiembaV2PublicBenchmark

benchmark_registry['movshon.FreemanZiemba2013.V1-pls'] = MovshonFreemanZiemba2013V1PLS
benchmark_registry['movshon.FreemanZiemba2013.V2-pls'] = MovshonFreemanZiemba2013V2PLS

# public benchmarks
benchmark_registry['movshon.FreemanZiemba2013public.V1-pls'] = FreemanZiembaV1PublicBenchmark
benchmark_registry['movshon.FreemanZiemba2013public.V2-pls'] = FreemanZiembaV2PublicBenchmark
