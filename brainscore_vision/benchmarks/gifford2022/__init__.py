from brainscore_vision import benchmark_registry
from .benchmark import Gifford2022

benchmark_registry['Gifford2022.IT-ridge'] = lambda: Gifford2022('IT', 'ridge')
benchmark_registry['Gifford2022.IT-ridgecv'] = lambda: Gifford2022('IT', 'ridgecv')