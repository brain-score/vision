# from brainscore_vision import benchmark_registry
# from .benchmark import
#
DATASETS = ['colour', 'contrast', 'cue-conflict', 'edge',
            'eidolonI', 'eidolonII', 'eidolonIII',
            'false-colour', 'high-pass', 'low-pass', 'phase-scrambling', 'power-equalisation',
            'rotation', 'silhouette', 'sketch', 'stylized', 'uniform-noise']
#
# for dataset in DATASETS:
#     # behavioral benchmark
#     identifier = f"Geirhos2021{dataset.replace('-', '')}ErrorConsistency"
#     globals()[identifier] = lambda dataset=dataset: _Geirhos2021ErrorConsistency(dataset)
#     # engineering benchmark
#     identifier = f"Geirhos2021{dataset.replace('-', '')}Accuracy"
#     globals()[identifier] = lambda dataset=dataset: _Geirhos2021Accuracy(dataset)
#


# Geirhos2021-error_consistency
from . import benchmark

for dataset in benchmark.DATASETS:
    assembly_identifier = f'Geirhos2021{dataset}'.replace('-', '')
    benchmark_ctr = getattr(benchmark, f"{assembly_identifier}ErrorConsistency")
    pool[f"brendel.{assembly_identifier}-error_consistency"] = LazyLoad(
        # use lambda parameter-binding to avoid `benchmark_ctr` being re-assigned in the next loop iteration
        lambda benchmark_ctr=benchmark_ctr: benchmark_ctr())


# Geirhos2021
for dataset in benchmark.DATASETS:
    assembly_identifier = f'Geirhos2021{dataset}'.replace('-', '')
    benchmark_ctr = getattr(benchmark, f"{assembly_identifier}Accuracy")
    pool[f"brendel.{assembly_identifier}-top1"] = LazyLoad(
        # use lambda parameter-binding to avoid `benchmark_ctr` being re-assigned in the next loop iteration
        lambda benchmark_ctr=benchmark_ctr: benchmark_ctr())