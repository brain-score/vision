from brainscore_vision import benchmark_registry

# neural predictivity
from .benchmark import SavaSegal2023_fMRI_pls

benchmark_registry['SavaSegal2023.fMRI-pls'] = SavaSegal2023_fMRI_pls
