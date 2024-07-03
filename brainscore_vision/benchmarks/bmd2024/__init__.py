from brainscore_vision import benchmark_registry
from brainscore_vision.benchmarks.bmd2024.benchmark import BMD2024AccuracyDistance

# behavioral benchmarks
benchmark_registry['BMD_2024_texture_1Behavioral-AccuracyDistance'] = lambda: BMD2024AccuracyDistance('texture_1')
benchmark_registry['BMD_2024_texture_2Behavioral-AccuracyDistance'] = lambda: BMD2024AccuracyDistance('texture_2')
benchmark_registry['BMD_2024_dotted_1Behavioral-AccuracyDistance'] = lambda: BMD2024AccuracyDistance('dotted_1')
benchmark_registry['BMD_2024_dotted_2Behavioral-AccuracyDistance'] = lambda: BMD2024AccuracyDistance('dotted_2')