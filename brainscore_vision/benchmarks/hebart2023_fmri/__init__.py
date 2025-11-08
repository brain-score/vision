from brainscore_vision import benchmark_registry
from .benchmark import Hebart2023fmri

benchmark_registry['Hebart2023_fmri.V1-ridge'] = lambda: Hebart2023fmri('V1', 'ridge')
benchmark_registry['Hebart2023_fmri.V2-ridge'] = lambda: Hebart2023fmri('V2', 'ridge')
benchmark_registry['Hebart2023_fmri.V4-ridge'] = lambda: Hebart2023fmri('V4', 'ridge')
benchmark_registry['Hebart2023_fmri.IT-ridge'] = lambda: Hebart2023fmri('IT', 'ridge')

benchmark_registry['Hebart2023_fmri.V1-ridgecv'] = lambda: Hebart2023fmri('V1', 'ridgecv')
benchmark_registry['Hebart2023_fmri.V2-ridgecv'] = lambda: Hebart2023fmri('V2', 'ridgecv')
benchmark_registry['Hebart2023_fmri.V4-ridgecv'] = lambda: Hebart2023fmri('V4', 'ridgecv')
benchmark_registry['Hebart2023_fmri.IT-ridgecv'] = lambda: Hebart2023fmri('IT', 'ridgecv')