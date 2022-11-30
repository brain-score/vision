from brainscore_vision import benchmark_registry
from .benchmark import MarquesDeValois1982V1PreferredOrientation

benchmark_registry['dicarlo.Marques2020_DeValois1982-pref_or'] = \
    MarquesDeValois1982V1PreferredOrientation
