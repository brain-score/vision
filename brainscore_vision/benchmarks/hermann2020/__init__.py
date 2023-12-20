from brainscore_vision import benchmark_registry
from .benchmark import Hermann2020cueconflictShapeBias, Hermann2020cueconflictShapeMatch

# invoke plugin tests

benchmark_registry['kornblith.Hermann2020cueconflict-shape_bias'] = Hermann2020cueconflictShapeBias
benchmark_registry['kornblith.Hermann2020cueconflict-shape_match'] = Hermann2020cueconflictShapeMatch
