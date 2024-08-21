from brainscore_vision import benchmark_registry
from . import benchmark

benchmark_registry['Lonnqvist2024-curves-20EngineeringAccuracy'] = lambda: benchmark._Lonnqvist2024EngineeringAccuracy('curves-20')