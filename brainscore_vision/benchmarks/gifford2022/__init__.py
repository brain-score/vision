from brainscore_vision import benchmark_registry
from .benchmark import Gifford2022

benchmark_registry['Gifford2022.IT-pls'] = lambda: Gifford2022('IT', 'pls')

benchmark_registry['Gifford2022.IT-ridge'] = lambda: Gifford2022('IT', 'ridge')

benchmark_registry['Gifford2022.IT-neuron_to_neuron'] = lambda: Gifford2022('IT', 'neuron_to_neuron')
