from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.data_helpers import s3


@pytest.mark.parametrize('benchmark', [
    'Scialom2024_rgbBehavioralAccuracyDistance',
    'Scialom2024_contoursBehavioralAccuracyDistance',
    'Scialom2024_phosphenes-12BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-16BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-21BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-27BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-35BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-46BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-59BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-77BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-100BehavioralAccuracyDistance',
    'Scialom2024_segments-12BehavioralAccuracyDistance',
    'Scialom2024_segments-16BehavioralAccuracyDistance',
    'Scialom2024_segments-21BehavioralAccuracyDistance',
    'Scialom2024_segments-27BehavioralAccuracyDistance',
    'Scialom2024_segments-35BehavioralAccuracyDistance',
    'Scialom2024_segments-46BehavioralAccuracyDistance',
    'Scialom2024_segments-59BehavioralAccuracyDistance',
    'Scialom2024_segments-77BehavioralAccuracyDistance',
    'Scialom2024_segments-100BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-allBehavioralErrorConsistency',
    'Scialom2024_segments-allBehavioralErrorConsistency'
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry


class TestBehavioral:
    def test_count(self):
        assert len(DATASETS) == 11 + 11 + 2  # phosphenes + segments + composites

    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('rgb', approx(0.98513, abs=0.001)),
        ('contours', approx(0.97848, abs=0.001)),
        ('phosphenes-12', approx(0.95416, abs=0.001)),
        ('phosphenes-16', approx(0.92583, abs=0.001)),
        ('phosphenes-21', approx(0.92166, abs=0.001)),
        ('phosphenes-27', approx(0.86888, abs=0.001)),
        ('phosphenes-35', approx(0.87277, abs=0.001)),
        ('phosphenes-46', approx(0.87125, abs=0.001)),
        ('phosphenes-59', approx(0.87625, abs=0.001)),
        ('phosphenes-77', approx(0.89277, abs=0.001)),
        ('phosphenes-100', approx(0.89930, abs=0.001)),
        ('segments-12', approx(0.89847, abs=0.001)),
        ('segments-16', approx(0.89055, abs=0.001)),
        ('segments-21', approx(0.88083, abs=0.001)),
        ('segments-27', approx(0.87083, abs=0.001)),
        ('segments-35', approx(0.86333, abs=0.001)),
        ('segments-46', approx(0.90250, abs=0.001)),
        ('segments-59', approx(0.87847, abs=0.001)),
        ('segments-77', approx(0.89013, abs=0.001)),
        ('segments-100', approx(0.93236, abs=0.001)),  # all of the above are AccuracyDistance
        ('phosphenes-all', approx(0.45755, abs=0.01)),  # alls are ErrorConsistency
        ('segments-all', approx(0.42529, abs=0.01)),
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        if 'all' in dataset:
            benchmark = f"Scialom2024_{dataset}BehavioralErrorConsistency"
        else:
            benchmark = f"Scialom2024_{dataset}BehavioralAccuracyDistance"
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark.ceiling
        assert ceiling == expected_ceiling

    @pytest.mark.parametrize('dataset, expected_raw_score', [
        ('rgb', approx(0.99000, abs=0.001)),
        ('contours', approx(0.53791, abs=0.001)),
        ('phosphenes-12', approx(0.87666, abs=0.001)),
        ('phosphenes-16', approx(0.83666, abs=0.001)),
        ('phosphenes-21', approx(0.83166, abs=0.001)),
        ('phosphenes-27', approx(0.73666, abs=0.001)),
        ('phosphenes-35', approx(0.68250, abs=0.001)),
        ('phosphenes-46', approx(0.59500, abs=0.001)),
        ('phosphenes-59', approx(0.50666, abs=0.001)),
        ('phosphenes-77', approx(0.42083, abs=0.001)),
        ('phosphenes-100', approx(0.31083, abs=0.001)),
        ('segments-12', approx(0.81500, abs=0.001)),
        ('segments-16', approx(0.73750, abs=0.001)),
        ('segments-21', approx(0.69666, abs=0.001)),
        ('segments-27', approx(0.63666, abs=0.001)),
        ('segments-35', approx(0.56833, abs=0.001)),
        ('segments-46', approx(0.48416, abs=0.001)),
        ('segments-59', approx(0.38750, abs=0.001)),
        ('segments-77', approx(0.28916, abs=0.001)),
        ('segments-100', approx(0.23916, abs=0.001)),  # all of the above are AccuracyDistance
        ('phosphenes-all', approx(0.18057, abs=0.01)),  # alls are ErrorConsistency
        ('segments-all', approx(0.15181, abs=0.01)),
    ])
    def test_model_8_degrees(self, dataset, expected_raw_score):
        if 'all' in dataset:
            benchmark = f"Scialom2024_{dataset}BehavioralErrorConsistency"
        else:
            benchmark = f"Scialom2024_{dataset}BehavioralAccuracyDistance"
        benchmark = load_benchmark(benchmark)
        filename = f"resnet50_julios_Scialom2024_{dataset}.nc"
        precomputed_features = Path(__file__).parent / filename
        s3.download_file_if_not_exists(precomputed_features,
                                       bucket='brainscore-vision', remote_filepath=f'/benchmarks/Scialom2024/{filename}')
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        score = benchmark(precomputed_features)
        raw_score = score.raw
        # division by ceiling <= 1 should result in higher score
        assert score >= raw_score
        assert raw_score == expected_raw_score


class TestEngineering:
    @pytest.mark.private_access
    @pytest.mark.parametrize('dataset, expected_accuracy', [
        ('rgb', approx(1.00000, abs=0.001)),
        ('contours', approx(0.52083, abs=0.001)),
        ('phosphenes-12', approx(0.08333, abs=0.001)),
        ('phosphenes-16', approx(0.08333, abs=0.001)),
        ('phosphenes-21', approx(0.08333, abs=0.001)),
        ('phosphenes-27', approx(0.08333, abs=0.001)),
        ('phosphenes-35', approx(0.08333, abs=0.001)),
        ('phosphenes-46', approx(0.08333, abs=0.001)),
        ('phosphenes-59', approx(0.08333, abs=0.001)),
        ('phosphenes-77', approx(0.08333, abs=0.001)),
        ('phosphenes-100', approx(0.06250, abs=0.001)),
        ('segments-12', approx(0.08333, abs=0.001)),
        ('segments-16', approx(0.08333, abs=0.001)),
        ('segments-21', approx(0.08333, abs=0.001)),
        ('segments-27', approx(0.12500, abs=0.001)),
        ('segments-35', approx(0.12500, abs=0.001)),
        ('segments-46', approx(0.14583, abs=0.001)),
        ('segments-59', approx(0.12500, abs=0.001)),
        ('segments-77', approx(0.10416, abs=0.001)),
        ('segments-100', approx(0.14583, abs=0.001))
    ])
    def test_accuracy(self, dataset, expected_accuracy):
        benchmark = load_benchmark(f"Scialom2024_{dataset}EngineeringAccuracy")
        filename = f"resnet50_julios_Scialom2024_{dataset}.nc"
        precomputed_features = Path(__file__).parent / filename
        s3.download_file_if_not_exists(precomputed_features,
                                       bucket='brainscore-vision', remote_filepath=f'/benchmarks/Scialom2024/{filename}')
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        score = benchmark(precomputed_features)
        raw_score = score.raw
        # division by ceiling <= 1 should result in higher score
        assert score >= raw_score
        assert raw_score == expected_accuracy
