from brainscore_vision import benchmark_registry
from .benchmark import (
    Li2026V1PLS, Li2026V2PLS, Li2026V4PLS, Li2026ITPLS,
    Li2026V1Ridge, Li2026V2Ridge, Li2026V4Ridge, Li2026ITRidge,
    Li2026V1Temporal, Li2026V2Temporal, Li2026V4Temporal, Li2026ITTemporal,
)

benchmark_registry['Li2026.V1-pls'] = Li2026V1PLS
benchmark_registry['Li2026.V2-pls'] = Li2026V2PLS
benchmark_registry['Li2026.V4-pls'] = Li2026V4PLS
benchmark_registry['Li2026.IT-pls'] = Li2026ITPLS

benchmark_registry['Li2026.V1-ridge'] = Li2026V1Ridge
benchmark_registry['Li2026.V2-ridge'] = Li2026V2Ridge
benchmark_registry['Li2026.V4-ridge'] = Li2026V4Ridge
benchmark_registry['Li2026.IT-ridge'] = Li2026ITRidge

benchmark_registry['Li2026.V1-temporal-pls'] = Li2026V1Temporal
benchmark_registry['Li2026.V2-temporal-pls'] = Li2026V2Temporal
benchmark_registry['Li2026.V4-temporal-pls'] = Li2026V4Temporal
benchmark_registry['Li2026.IT-temporal-pls'] = Li2026ITTemporal
