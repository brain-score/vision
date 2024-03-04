from brainscore_vision import benchmark_registry
from . import benchmark

DATASETS = ['rgb', 'contours', 'phosphenes-12', 'phosphenes-16', 'phosphenes-21', 'phosphenes-27', 'phosphenes-35',
            'phosphenes-46', 'phosphenes-59', 'phosphenes-77', 'phosphenes-100', 'segments-12', 'segments-16',
            'segments-21', 'segments-27', 'segments-35', 'segments-46', 'segments-59', 'segments-77', 'segments-100']


for dataset in DATASETS:
    # condition-specific benchmarks with a coarse-grained metric
    benchmark_registry[f"Scialom2024_{dataset}AccuracyDistance"] = benchmark._Scialom2024BehavioralAccuracyDistance(dataset)
    # engineering benchmark
    benchmark_registry[f"Scialom2024_{dataset}EngineeringAccuracy"] = benchmark._Scialom2024EngineeringAccuracy(dataset)

# composite benchmark with a fine-grained metric
benchmark_registry[f"Scialom2024_phosphenes-compositeErrorConsistency"] = benchmark._Scialom2024ErrorConsistency('phosphenes-composite')
benchmark_registry[f"Scialom2024_segments-compositeErrorConsistency"] = benchmark._Scialom2024ErrorConsistency('segments-composite')
