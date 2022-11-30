from brainscore_vision import benchmark_registry
from .benchmark import DicarloSanghaviJozwik2020V4PLS, DicarloSanghaviJozwik2020ITPLS

benchmark_registry['dicarlo.SanghaviJozwik2020.V4-pls'] = DicarloSanghaviJozwik2020V4PLS
benchmark_registry['dicarlo.SanghaviJozwik2020.IT-pls'] = DicarloSanghaviJozwik2020ITPLS
