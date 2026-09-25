from brainscore_vision import benchmark_registry
from . import benchmark

DATASETS = ['Ori', 'Fil', 'Lin', 'SilTex', 'Sil', 'Tex', 'LinFil', 'Sparse', 'Dense']

benchmark_registry['Kato2026Ori-error_consistency'] = getattr(benchmark, "Kato2026OriErrorConsistency")
benchmark_registry['Kato2026Fil-error_consistency'] = getattr(benchmark, "Kato2026FilErrorConsistency")
benchmark_registry['Kato2026Lin-error_consistency'] = getattr(benchmark, "Kato2026LinErrorConsistency")
benchmark_registry['Kato2026SilTex-error_consistency'] = getattr(benchmark, "Kato2026SilTexErrorConsistency")
benchmark_registry['Kato2026Sil-error_consistency'] = getattr(benchmark, "Kato2026SilErrorConsistency")
benchmark_registry['Kato2026Tex-error_consistency'] = getattr(benchmark, "Kato2026TexErrorConsistency")
benchmark_registry['Kato2026LinFil-error_consistency'] = getattr(benchmark, "Kato2026LinFilErrorConsistency")
benchmark_registry['Kato2026Sparse-error_consistency'] = getattr(benchmark, "Kato2026SparseErrorConsistency")
benchmark_registry['Kato2026Dense-error_consistency'] = getattr(benchmark, "Kato2026DenseErrorConsistency")

benchmark_registry['Kato2026Ori-accuracy_distance'] = getattr(benchmark, "Kato2026OriAccuracyDistance")
benchmark_registry['Kato2026Fil-accuracy_distance'] = getattr(benchmark, "Kato2026FilAccuracyDistance")
benchmark_registry['Kato2026Lin-accuracy_distance'] = getattr(benchmark, "Kato2026LinAccuracyDistance")
benchmark_registry['Kato2026SilTex-accuracy_distance'] = getattr(benchmark, "Kato2026SilTexAccuracyDistance")
benchmark_registry['Kato2026Sil-accuracy_distance'] = getattr(benchmark, "Kato2026SilAccuracyDistance")
benchmark_registry['Kato2026Tex-accuracy_distance'] = getattr(benchmark, "Kato2026TexAccuracyDistance")
benchmark_registry['Kato2026LinFil-accuracy_distance'] = getattr(benchmark, "Kato2026LinFilAccuracyDistance")
benchmark_registry['Kato2026Sparse-accuracy_distance'] = getattr(benchmark, "Kato2026SparseAccuracyDistance")
benchmark_registry['Kato2026Dense-accuracy_distance'] = getattr(benchmark, "Kato2026DenseAccuracyDistance")
