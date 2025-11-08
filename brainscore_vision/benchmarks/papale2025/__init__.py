from brainscore_vision import benchmark_registry
from .benchmark import Papale2025

benchmark_registry['Papale2025.V1-ridge'] = lambda: Papale2025('V1', 'ridge')
benchmark_registry['Papale2025.V4-ridge'] = lambda: Papale2025('V4', 'ridge')
benchmark_registry['Papale2025.IT-ridge'] = lambda: Papale2025('IT', 'ridge')

benchmark_registry['Papale2025.V1-ridgecv'] = lambda: Papale2025('V1', 'ridgecv')
benchmark_registry['Papale2025.V4-ridgecv'] = lambda: Papale2025('V4', 'ridgecv')
benchmark_registry['Papale2025.IT-ridgecv'] = lambda: Papale2025('IT', 'ridgecv')

