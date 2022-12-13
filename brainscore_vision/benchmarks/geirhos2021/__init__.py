from brainscore_vision import benchmark_registry
from . import benchmark

DATASETS = ['colour', 'contrast', 'cue-conflict', 'edge',
            'eidolonI', 'eidolonII', 'eidolonIII',
            'false-colour', 'high-pass', 'low-pass', 'phase-scrambling', 'power-equalisation',
            'rotation', 'silhouette', 'sketch', 'stylized', 'uniform-noise']

# Geirhos2021-error_consistency
for dataset in benchmark.DATASETS:
    assembly_identifier = f'Geirhos2021{dataset}'.replace('-', '')
    benchmark_ctr = getattr(benchmark, f"{assembly_identifier}ErrorConsistency")
    # use lambda parameter-binding to avoid `benchmark_ctr` being re-assigned in the next loop iteration
    benchmark_registry[f"brendel.{assembly_identifier}-error_consistency"] = \
        lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()


# Geirhos2021
for dataset in benchmark.DATASETS:
    assembly_identifier = f'Geirhos2021{dataset}'.replace('-', '')
    benchmark_ctr = getattr(benchmark, f"{assembly_identifier}Accuracy")
    benchmark_registry[f"brendel.{assembly_identifier}-top1"] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
