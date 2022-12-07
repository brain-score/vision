from brainscore_vision import benchmark_registry
from .benchmark import MovshonFreemanZiemba2013V1PLS, MovshonFreemanZiemba2013V2PLS, MovshonFreemanZiemba2013V1RDM, \
    MovshonFreemanZiemba2013V2RDM, MovshonFreemanZiemba2013V1Single

benchmark_registry['movshon.FreemanZiemba2013.V1-pls'] = MovshonFreemanZiemba2013V1PLS
benchmark_registry['movshon.FreemanZiemba2013.V2-pls'] = MovshonFreemanZiemba2013V2PLS

benchmark_registry['movshon.FreemanZiemba2013.V1-rdm'] = MovshonFreemanZiemba2013V1RDM
benchmark_registry['movshon.FreemanZiemba2013.V2-rdm'] = MovshonFreemanZiemba2013V2RDM
benchmark_registry['movshon.FreemanZiemba2013.V1-single'] = MovshonFreemanZiemba2013V1Single
