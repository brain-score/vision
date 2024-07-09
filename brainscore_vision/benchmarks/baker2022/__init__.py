from brainscore_vision import benchmark_registry
from brainscore_vision.benchmarks.baker2022.benchmark import Baker2022AccuracyDeltaFrankenstein, \
    Baker2022AccuracyDeltaFragmented, Baker2022InvertedAccuracyDelta

DATASETS = ['normal', 'inverted']

benchmark_registry['Baker2022frankenstein-accuracy_delta'] = lambda: Baker2022AccuracyDeltaFrankenstein()
benchmark_registry['Baker2022fragmented-accuracy_delta'] = lambda: Baker2022AccuracyDeltaFragmented()
benchmark_registry['Baker2022inverted-accuracy_delta'] = lambda: Baker2022InvertedAccuracyDelta()
