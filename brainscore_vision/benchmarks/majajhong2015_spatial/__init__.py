# developed in Schrimpf et al. 2024 https://www.biorxiv.org/content/10.1101/2024.01.09.572970

from brainscore_vision import benchmark_registry

from .benchmark import DicarloMajajHong2015ITSpatialCorrelation

benchmark_registry['MajajHong2015.IT-spatial_correlation'] = DicarloMajajHong2015ITSpatialCorrelation
