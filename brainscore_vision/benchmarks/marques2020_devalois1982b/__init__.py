from brainscore_vision import benchmark_registry
from .benchmark import MarquesDeValois1982V1PeakSpatialFrequency

benchmark_registry['dicarlo.Marques2020_DeValois1982-peak_sf'] = \
    MarquesDeValois1982V1PeakSpatialFrequency
