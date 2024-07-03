from brainscore_vision import benchmark_registry
from brainscore_vision.benchmarks.BMD_2024.benchmark import BMD2024AccuracyDistance

# behavioral benchmarks
benchmark_registry['BMD_2024_texture_1BehavioralAccuracyDistance'] = lambda: BMD2024AccuracyDistance('texture_1')
benchmark_registry['BMD_2024_texture_2BehavioralAccuracyDistance'] = lambda: BMD2024AccuracyDistance('texture_2')
benchmark_registry['BMD_2024_dotted_1BehavioralAccuracyDistance'] = lambda: BMD2024AccuracyDistance('dotted_1')
benchmark_registry['BMD_2024_dotted_2BehavioralAccuracyDistance'] = lambda: BMD2024AccuracyDistance('dotted_2')