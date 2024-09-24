from brainscore_vision import benchmark_registry

# neural predictivity
from .benchmark import DicarloMajajHong2015V4PLS, DicarloMajajHong2015ITPLS

benchmark_registry['MajajHong2015.V4-pls'] = DicarloMajajHong2015V4PLS
benchmark_registry['MajajHong2015.IT-pls'] = DicarloMajajHong2015ITPLS

# public
from .benchmark import MajajHongV4PublicBenchmark, MajajHongITPublicBenchmark

benchmark_registry['MajajHong2015public.V4-pls'] = MajajHongV4PublicBenchmark
benchmark_registry['MajajHong2015public.IT-pls'] = MajajHongITPublicBenchmark

# temporal
from .benchmark import MajajHongV4TemporalPublicBenchmark, MajajHongITTemporalPublicBenchmark
benchmark_registry['MajajHong2015public.V4-temporal-pls'] = lambda: MajajHongV4TemporalPublicBenchmark(time_interval=10)
benchmark_registry['MajajHong2015public.IT-temporal-pls'] = lambda: MajajHongITTemporalPublicBenchmark(time_interval=10)
