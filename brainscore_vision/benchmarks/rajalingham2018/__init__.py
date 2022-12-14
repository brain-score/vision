from brainscore_vision import benchmark_registry
from .benchmarks.benchmark import DicarloRajalingham2018I2n

benchmark_registry['dicarlo.Rajalingham2018-i2n'] = DicarloRajalingham2018I2n


# public benchmark
from.benchmarks.public_benchmark import RajalinghamMatchtosamplePublicBenchmark
