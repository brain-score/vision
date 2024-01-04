from brainscore_vision import benchmark_registry
from .benchmarks.sanghavi2020_benchmark import DicarloSanghavi2020V4PLS, DicarloSanghavi2020ITPLS

benchmark_registry['Sanghavi2020.V4-pls'] = DicarloSanghavi2020V4PLS
benchmark_registry['Sanghavi2020.IT-pls'] = DicarloSanghavi2020ITPLS


from .benchmarks.sanghavijozwik2020_benchmark import DicarloSanghaviJozwik2020V4PLS, DicarloSanghaviJozwik2020ITPLS

benchmark_registry['SanghaviJozwik2020.V4-pls'] = DicarloSanghaviJozwik2020V4PLS
benchmark_registry['SanghaviJozwik2020.IT-pls'] = DicarloSanghaviJozwik2020ITPLS


from .benchmarks.sanghavimurty2020_benchmark import DicarloSanghaviMurty2020V4PLS, DicarloSanghaviMurty2020ITPLS

benchmark_registry['SanghaviMurty2020.V4-pls'] = DicarloSanghaviMurty2020V4PLS
benchmark_registry['SanghaviMurty2020.IT-pls'] = DicarloSanghaviMurty2020ITPLS
