from brainscore_vision import benchmark_registry
from .benchmark import MarquesSchiller1976V1SpatialFrequencyBandwidth, \
        MarquesSchiller1976V1SpatialFrequencySelective

# V1 properties benchmarks: spatial frequency
benchmark_registry['dicarlo.Marques2020_Schiller1976-sf_selective'] = MarquesSchiller1976V1SpatialFrequencySelective
benchmark_registry['dicarlo.Marques2020_Schiller1976-sf_bandwidth'] = MarquesSchiller1976V1SpatialFrequencyBandwidth
