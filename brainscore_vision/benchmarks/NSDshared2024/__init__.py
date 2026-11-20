from brainscore_vision import benchmark_registry

# neural predictivity
from .benchmark import NSDV1SharedPLS, NSDV2SharedPLS, NSDV3SharedPLS, NSDV4SharedPLS
from .benchmark import NSDEarlySharedPLS, NSDLateralSharedPLS, NSDParietalSharedPLS, NSDVentralSharedPLS

benchmark_registry['NSD.V1.pls'] = NSDV1SharedPLS
benchmark_registry['NSD.V2.pls'] = NSDV2SharedPLS
benchmark_registry['NSD.V3.pls'] = NSDV3SharedPLS
benchmark_registry['NSD.V4.pls'] = NSDV4SharedPLS
benchmark_registry['NSD.early.pls'] = NSDEarlySharedPLS
benchmark_registry['NSD.lateral.pls'] = NSDLateralSharedPLS
benchmark_registry['NSD.parietal.pls'] = NSDParietalSharedPLS
benchmark_registry['NSD.ventral.pls'] = NSDVentralSharedPLS

