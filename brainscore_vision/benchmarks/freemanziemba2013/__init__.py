from brainscore_vision import benchmark_registry
from .benchmarks.benchmark import MovshonFreemanZiemba2013V1PLS, MovshonFreemanZiemba2013V2PLS, MovshonFreemanZiemba2013V1RDM, \
    MovshonFreemanZiemba2013V2RDM, MovshonFreemanZiemba2013V1Single
from .benchmarks.public_benchmarks import FreemanZiembaV1PublicBenchmark, FreemanZiembaV2PublicBenchmark

benchmark_registry['movshon.FreemanZiemba2013.V1-pls'] = MovshonFreemanZiemba2013V1PLS
benchmark_registry['movshon.FreemanZiemba2013.V2-pls'] = MovshonFreemanZiemba2013V2PLS

benchmark_registry['movshon.FreemanZiemba2013.V1-rdm'] = MovshonFreemanZiemba2013V1RDM
benchmark_registry['movshon.FreemanZiemba2013.V2-rdm'] = MovshonFreemanZiemba2013V2RDM
benchmark_registry['movshon.FreemanZiemba2013.V1-single'] = MovshonFreemanZiemba2013V1Single

# public benchmarks
benchmark_registry['movshon.FreemanZiemba2013.V1.public'] = FreemanZiembaV1PublicBenchmark
benchmark_registry['movshon.FreemanZiemba2013.V2.public'] = FreemanZiembaV2PublicBenchmark
