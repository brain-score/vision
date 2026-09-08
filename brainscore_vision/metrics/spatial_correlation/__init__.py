# developed in Schrimpf et al. 2024 https://www.biorxiv.org/content/10.1101/2024.01.09.572970

from brainscore_vision import metric_registry
from .metric import SpatialCorrelationSimilarity

metric_registry['spatial_correlation'] = SpatialCorrelationSimilarity
