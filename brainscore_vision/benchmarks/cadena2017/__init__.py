from brainscore_vision import benchmark_registry
from .benchmark import ToliasCadena2017PLS, ToliasCadena2017Mask

benchmark_registry['Cadena2017-pls'] = ToliasCadena2017PLS
benchmark_registry['Cadena2017-mask'] = ToliasCadena2017Mask
