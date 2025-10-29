from brainscore_vision import benchmark_registry
from .benchmark import Hebart2023fmri

benchmark_registry['Hebart2023_fmri.V1-pls'] = lambda: Hebart2023fmri('V1', 'pls')
benchmark_registry['Hebart2023_fmri.V2-pls'] = lambda: Hebart2023fmri('V2', 'pls')
benchmark_registry['Hebart2023_fmri.V4-pls'] = lambda: Hebart2023fmri('V4', 'pls')
benchmark_registry['Hebart2023_fmri.IT-pls'] = lambda: Hebart2023fmri('IT', 'pls')

benchmark_registry['Hebart2023_fmri.V1-ridge'] = lambda: Hebart2023fmri('V1', 'ridge')
benchmark_registry['Hebart2023_fmri.V2-ridge'] = lambda: Hebart2023fmri('V2', 'ridge')
benchmark_registry['Hebart2023_fmri.V4-ridge'] = lambda: Hebart2023fmri('V4', 'ridge')
benchmark_registry['Hebart2023_fmri.IT-ridge'] = lambda: Hebart2023fmri('IT', 'ridge')

benchmark_registry['Hebart2023_fmri.V1-neuron_to_neuron'] = lambda: Hebart2023fmri('V1', 'neuron_to_neuron')
benchmark_registry['Hebart2023_fmri.V2-neuron_to_neuron'] = lambda: Hebart2023fmri('V2', 'neuron_to_neuron')
benchmark_registry['Hebart2023_fmri.V4-neuron_to_neuron'] = lambda: Hebart2023fmri('V4', 'neuron_to_neuron')
benchmark_registry['Hebart2023_fmri.IT-neuron_to_neuron'] = lambda: Hebart2023fmri('IT', 'neuron_to_neuron')
