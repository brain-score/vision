from brainscore_vision import benchmark_registry
from brainscore_vision.benchmarks.baker2022.benchmark import Baker2022AccuracyDeltaFrankenstein, \
    Baker2022AccuracyDeltaFragmented, Baker2022InvertedAccuracyDelta

benchmark_registry['aker2022-accuracy_delta_frankenstein'] = lambda: Baker2022AccuracyDeltaFrankenstein()
benchmark_registry['Baker2022-accuracy_delta_fragmented'] = lambda: Baker2022AccuracyDeltaFragmented()
benchmark_registry['Baker2022-inverted_accuracy_delta'] = lambda: Baker2022InvertedAccuracyDelta()
