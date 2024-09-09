from brainscore_vision import benchmark_registry

# neural predictivity
from .benchmark import NSDV1SharedPLS

benchmark_registry['NSD.V1.pls'] = NSDV1SharedPLS
