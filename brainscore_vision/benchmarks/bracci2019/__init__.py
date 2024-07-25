from brainscore_vision import benchmark_registry
from .benchmark import Bracci2019RSA

# benchmark_registry['Bracci2019_RSA-V1'] = lambda: _Bracci2019RSA("V1")
# benchmark_registry['Bracci2019_RSA-posteriorVTC'] = lambda: _Bracci2019RSA("posteriorVTC")
benchmark_registry['Bracci2019.anteriorVTC-rdm'] = lambda: Bracci2019RSA("anteriorVTC")


