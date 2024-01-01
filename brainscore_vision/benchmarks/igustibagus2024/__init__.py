from brainscore_vision import benchmark_registry

# neural alignment
from .domain_transfer_neural import Igustibagus2024_ridge

benchmark_registry['Igustibagus2024-ridge'] = Igustibagus2024_ridge

# analysis benchmarks
from .domain_transfer_analysis import OOD_AnalysisBenchmark

benchmark_registry['Igustibagus2024.IT_readout-accuracy'] = OOD_AnalysisBenchmark
