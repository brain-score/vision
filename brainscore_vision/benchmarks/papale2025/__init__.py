from brainscore_vision import benchmark_registry
from .benchmark import Papale2025PLS, Papale2025Ridge, Papale2025NeuronToNeuron

benchmark_registry['Papale2025.V1-pls'] = lambda: Papale2025PLS('V1')
benchmark_registry['Papale2025.V4-pls'] = lambda: Papale2025PLS('V4')
benchmark_registry['Papale2025.IT-pls'] = lambda: Papale2025PLS('IT')

benchmark_registry['Papale2025.V1-ridge'] = lambda: Papale2025Ridge('V1')
benchmark_registry['Papale2025.V4-ridge'] = lambda: Papale2025Ridge('V4')
benchmark_registry['Papale2025.IT-ridge'] = lambda: Papale2025Ridge('IT')

benchmark_registry['Papale2025.V1-neuron_to_neuron'] = lambda: Papale2025NeuronToNeuron('V1')
benchmark_registry['Papale2025.V4-neuron_to_neuron'] = lambda: Papale2025NeuronToNeuron('V4')
benchmark_registry['Papale2025.IT-neuron_to_neuron'] = lambda: Papale2025NeuronToNeuron('IT')

