from brainscore_vision import benchmark_registry
from .benchmark import MajajHongV4PublicBenchmark, MajajHongITPublicBenchmark

benchmark_registry['MajajHong2015public.V4-reverse_pls'] = MajajHongV4PublicBenchmark
benchmark_registry['MajajHong2015public.IT-reverse_pls'] = MajajHongITPublicBenchmark

