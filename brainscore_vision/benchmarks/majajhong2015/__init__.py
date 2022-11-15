# neural predictivity
from .benchmark import DicarloMajajHong2015V4PLS, DicarloMajajHong2015ITPLS

benchmark_registry['dicarlo.MajajHong2015.V4-pls'] = DicarloMajajHong2015V4PLS
benchmark_registry['dicarlo.MajajHong2015.IT-pls'] = DicarloMajajHong2015ITPLS

# experimental
from .benchmark import DicarloMajajHong2015V4Mask, DicarloMajajHong2015ITMask, \
    DicarloMajajHong2015V4RDM, DicarloMajajHong2015ITRDM

benchmark_registry['dicarlo.MajajHong2015.V4-mask'] = DicarloMajajHong2015V4Mask
benchmark_registry['dicarlo.MajajHong2015.IT-mask'] = DicarloMajajHong2015ITMask
benchmark_registry['dicarlo.MajajHong2015.V4-rdm'] = DicarloMajajHong2015V4RDM
benchmark_registry['dicarlo.MajajHong2015.IT-rdm'] = DicarloMajajHong2015ITRDM

# public
from .benchmark import MajajHongV4PublicBenchmark, MajajHongITPublicBenchmark

benchmark_registry['dicarlo.MajajHong2015public.V4-pls'] = MajajHongV4PublicBenchmark
benchmark_registry['dicarlo.MajajHong2015public.IT-pls'] = MajajHongITPublicBenchmark
