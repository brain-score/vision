from brainscore_vision import benchmark_registry
from brainscore_vision.benchmarks.bmd2024.benchmark import BMD2024AccuracyDistance

# behavioral benchmarks
benchmark_registry['BMD2024.texture_1Behavioral-accuracy_distance'] = lambda: BMD2024AccuracyDistance('texture_1')
benchmark_registry['BMD2024.texture_2Behavioral-accuracy_distance'] = lambda: BMD2024AccuracyDistance('texture_2')
benchmark_registry['BMD2024.dotted_1Behavioral-accuracy_distance'] = lambda: BMD2024AccuracyDistance('dotted_1')
benchmark_registry['BMD2024.dotted_2Behavioral-accuracy_distance'] = lambda: BMD2024AccuracyDistance('dotted_2')
