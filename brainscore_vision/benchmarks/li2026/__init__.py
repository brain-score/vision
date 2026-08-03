from brainscore_vision import benchmark_registry
from .benchmark import (
    Li2026V1RidgeCV, Li2026V2RidgeCV, Li2026V4RidgeCV, Li2026ITRidgeCV,
    Li2026V1PLS, Li2026V2PLS, Li2026V4PLS, Li2026ITPLS,
)

benchmark_registry['Li2026.V1-ridgecv'] = Li2026V1RidgeCV
benchmark_registry['Li2026.V2-ridgecv'] = Li2026V2RidgeCV
benchmark_registry['Li2026.V4-ridgecv'] = Li2026V4RidgeCV
benchmark_registry['Li2026.IT-ridgecv'] = Li2026ITRidgeCV

benchmark_registry['Li2026.V1-pls'] = Li2026V1PLS
benchmark_registry['Li2026.V2-pls'] = Li2026V2PLS
benchmark_registry['Li2026.V4-pls'] = Li2026V4PLS
benchmark_registry['Li2026.IT-pls'] = Li2026ITPLS
