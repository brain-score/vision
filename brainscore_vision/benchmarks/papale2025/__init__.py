from brainscore_vision import benchmark_registry
from .benchmark import Papale2025

benchmark_registry['Papale2025.V1-pls'] = lambda: Papale2025('V1', 'pls')
benchmark_registry['Papale2025.V4-pls'] = lambda: Papale2025('V4', 'pls')
benchmark_registry['Papale2025.IT-pls'] = lambda: Papale2025('IT', 'pls')

benchmark_registry['Papale2025.V1-ridge'] = lambda: Papale2025('V1', 'ridge')
benchmark_registry['Papale2025.V4-ridge'] = lambda: Papale2025('V4', 'ridge')
benchmark_registry['Papale2025.IT-ridge'] = lambda: Papale2025('IT', 'ridge')

benchmark_registry['Papale2025.V1-neuron_to_neuron'] = lambda: Papale2025('V1', 'neuron_to_neuron')
benchmark_registry['Papale2025.V4-neuron_to_neuron'] = lambda: Papale2025('V4', 'neuron_to_neuron')
benchmark_registry['Papale2025.IT-neuron_to_neuron'] = lambda: Papale2025('IT', 'neuron_to_neuron')

