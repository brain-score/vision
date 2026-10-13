from brainscore_vision import benchmark_registry
from .benchmark import (
    Li2026V1RidgeCV, Li2026V2RidgeCV, Li2026V4RidgeCV, Li2026ITRidgeCV,
    Li2026V1RDM, Li2026V2RDM, Li2026V4RDM, Li2026ITRDM,
    Li2026V1PLS, Li2026V2PLS, Li2026V4PLS, Li2026ITPLS,
    Li2026V1Temporal, Li2026V2Temporal, Li2026V4Temporal, Li2026ITTemporal,
)

benchmark_registry['Li2026.V1-ridgecv'] = Li2026V1RidgeCV
benchmark_registry['Li2026.V2-ridgecv'] = Li2026V2RidgeCV
benchmark_registry['Li2026.V4-ridgecv'] = Li2026V4RidgeCV
benchmark_registry['Li2026.IT-ridgecv'] = Li2026ITRidgeCV

benchmark_registry['Li2026.V1-rdm'] = Li2026V1RDM
benchmark_registry['Li2026.V2-rdm'] = Li2026V2RDM
benchmark_registry['Li2026.V4-rdm'] = Li2026V4RDM
benchmark_registry['Li2026.IT-rdm'] = Li2026ITRDM

benchmark_registry['Li2026.V1-pls'] = Li2026V1PLS
benchmark_registry['Li2026.V2-pls'] = Li2026V2PLS
benchmark_registry['Li2026.V4-pls'] = Li2026V4PLS
benchmark_registry['Li2026.IT-pls'] = Li2026ITPLS

# Offline use only: unparented, kept out of scoring runs (see _Li2026RegionTemporal).
benchmark_registry['Li2026.V1-temporal-pls'] = Li2026V1Temporal
benchmark_registry['Li2026.V2-temporal-pls'] = Li2026V2Temporal
benchmark_registry['Li2026.V4-temporal-pls'] = Li2026V4Temporal
benchmark_registry['Li2026.IT-temporal-pls'] = Li2026ITTemporal
