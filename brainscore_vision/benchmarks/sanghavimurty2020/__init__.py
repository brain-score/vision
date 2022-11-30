from brainscore_vision import benchmark_registry
from .benchmark import DicarloSanghaviMurty2020V4PLS, DicarloSanghaviMurty2020ITPLS

benchmark_registry['dicarlo.SanghaviMurty2020.V4-pls'] = DicarloSanghaviMurty2020V4PLS
benchmark_registry['dicarlo.SanghaviMurty2020.IT-pls'] = DicarloSanghaviMurty2020ITPLS
