import pytest
from pytest import approx
from brainscore_vision import benchmark_registry, load_benchmark, load_model
from brainscore_vision.benchmarks.kato2026.benchmark import DATASETS

@pytest.mark.parametrize('benchmark', [
    'Kato2026Ori-error_consistency',
    'Kato2026Fil-error_consistency',
    'Kato2026Lin-error_consistency',
    'Kato2026SilTex-error_consistency',
    'Kato2026Sil-error_consistency',
    'Kato2026Tex-error_consistency',
    'Kato2026LinFil-error_consistency',
    'Kato2026Sparse-error_consistency',
    'Kato2026Dense-error_consistency',
    'Kato2026Ori-accuracy_distance',
    'Kato2026Fil-accuracy_distance',
    'Kato2026Lin-accuracy_distance',
    'Kato2026SilTex-accuracy_distance',
    'Kato2026Sil-accuracy_distance',    
    'Kato2026Tex-accuracy_distance',
    'Kato2026LinFil-accuracy_distance',
    'Kato2026Sparse-accuracy_distance',
    'Kato2026Dense-accuracy_distance',   
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry

class TestErrorConsistency:
    def test_count(self):
        assert len(DATASETS) == 9

    @pytest.mark.parametrize('dataset', DATASETS)
    def test_in_pool(self, dataset):
        identifier = f"Kato2026{dataset.replace('-', '')}-error_consistency"
        assert identifier in benchmark_registry

    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('Ori', approx(0.21271, abs=0.0001)),
        ('Fil', approx(0.34131, abs=0.0001)),
        ('Lin', approx(0.33968, abs=0.0001)),
        ('SilTex', approx(0.43446, abs=0.0001)),
        ('Sil', approx(0.52434, abs=0.0001)),
        ('Tex', approx(0.32948, abs=0.0001)),
        ('LinFil', approx(0.43865, abs=0.0001)),
        ('Sparse', approx(0.39401, abs=0.0001)),
        ('Dense', approx(0.35777, abs=0.0001)),
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        benchmark = f"Kato2026{dataset.replace('-', '')}-error_consistency"
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark.ceiling
        assert ceiling == expected_ceiling

    @pytest.mark.parametrize('dataset, expected_raw_score', [
        ('Ori', approx(0.04396, abs=0.0001)),
        ('Fil', approx(0.15168, abs=0.0001)),
        ('Lin', approx(0.05427, abs=0.0001)),
        ('SilTex', approx(0.04693, abs=0.0001)),
        ('Sil', approx(0.06757, abs=0.0001)),
        ('Tex', approx(0.18094, abs=0.0001)),
        ('LinFil', approx(0.00906, abs=0.0001)),
        ('Sparse', approx(0.00000, abs=0.0001)),
        ('Dense', approx(0.00000, abs=0.0001)),
    ])
    def test_model_alexnet(self, dataset, expected_raw_score):
        benchmark = load_benchmark(f"Kato2026{dataset.replace('-', '')}-error_consistency")
        model = load_model('alexnet')
        score = benchmark(model)
        raw_score = score.raw
        # division by ceiling <= 1 should result in higher score
        assert score >= raw_score
        assert raw_score == expected_raw_score
        
class TestAccuracyDistance:
    def test_count(self):
        assert len(DATASETS) == 9

    @pytest.mark.parametrize('dataset', DATASETS)
    def test_in_pool(self, dataset):
        identifier = f"Kato2026{dataset.replace('-', '')}-accuracy_distance"
        assert identifier in benchmark_registry

    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('Ori', approx(0.88677, abs=0.0001)),
        ('Fil', approx(0.90117, abs=0.0001)),
        ('Lin', approx(0.94708, abs=0.0001)),
        ('SilTex', approx(0.94806, abs=0.0001)),
        ('Sil', approx(0.92978, abs=0.0001)),
        ('Tex', approx(0.93304, abs=0.0001)),
        ('LinFil', approx(0.91169, abs=0.0001)),
        ('Sparse', approx(0.77562, abs=0.0001)),
        ('Dense', approx(0.81783, abs=0.0001)),
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        benchmark = f"Kato2026{dataset.replace('-', '')}-accuracy_distance"
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark.ceiling
        assert ceiling == expected_ceiling

    @pytest.mark.parametrize('dataset, expected_raw_score', [
        ('Ori', approx(0.77458, abs=0.0001)), 
        ('Fil', approx(0.75369, abs=0.0001)),
        ('Lin', approx(0.20051, abs=0.0001)),
        ('SilTex', approx(0.46979, abs=0.0001)),
        ('Sil', approx(0.09714, abs=0.0001)),
        ('Tex', approx(0.94172, abs=0.0001)),
        ('LinFil', approx(0.54994, abs=0.0001)),
        ('Sparse', approx(0.04, abs=0.0001)),
        ('Dense', approx(0.04, abs=0.0001)),
    ])
    def test_model_alexnet(self, dataset, expected_raw_score):
        benchmark = load_benchmark(f"Kato2026{dataset.replace('-', '')}-accuracy_distance")
        model = load_model('alexnet')
        score = benchmark(model)
        raw_score = score.raw
        # division by ceiling <= 1 should result in higher score
        assert score >= raw_score
        assert raw_score == expected_raw_score