from brainscore_vision import benchmark_registry

# Malania2007

from . import benchmark

for dataset in benchmark.DATASETS:
    assembly_identifier = f"Malania2007_{dataset}"
    benchmark_ctr = getattr(benchmark, f"{assembly_identifier}")
    benchmark_registry[f"{assembly_identifier}"] = benchmark_ctr
