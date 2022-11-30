from brainscore_vision import benchmark_registry
from .benchmark import DicarloSanghavi2020V4PLS, DicarloSanghavi2020ITPLS

benchmark_registry['dicarlo.Sanghavi2020.V4-pls'] = DicarloSanghavi2020V4PLS
benchmark_registry['dicarlo.Sanghavi2020.IT-pls'] = DicarloSanghavi2020ITPLS
