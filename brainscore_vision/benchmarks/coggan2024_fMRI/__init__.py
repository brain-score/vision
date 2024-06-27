# Created by David Coggan on 2024 06 25

from brainscore_vision import benchmark_registry
from .benchmark import (
    Coggan2024_V1,
    Coggan2024_V2,
    Coggan2024_V4,
    Coggan2024_IT,
)

benchmark_registry['Coggan2024_fMRI.V1'] = Coggan2024_V1
benchmark_registry['Coggan2024_fMRI.V2'] = Coggan2024_V2
benchmark_registry['Coggan2024_fMRI.V4'] = Coggan2024_V4
benchmark_registry['Coggan2024_fMRI.IT'] = Coggan2024_IT

