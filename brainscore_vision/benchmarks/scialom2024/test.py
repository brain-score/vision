from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.benchmarks.scialom2024 import DATASETS
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
    'Scialom2024_phosphenes-compositeBehavioralErrorConsistency',
    'Scialom2024_segments-compositeBehavioralErrorConsistency'
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry


class TestBehavioral:
    def test_count(self):
        assert len(DATASETS) == 11 + 11 + 2  # phosphenes + segments + composites

    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('rgb', approx(0.98034, abs=0.001)),
        ('contours', approx(0.9682, abs=0.001)),
        ('phosphenes-12', approx(0.71819, abs=0.001)),
        ('phosphenes-16', approx(0.67069, abs=0.001)),
        ('phosphenes-21', approx(0.68930, abs=0.001)),
        ('phosphenes-27', approx(0.62750, abs=0.001)),
        ('phosphenes-35', approx(0.61458, abs=0.001)),
        ('phosphenes-46', approx(0.62916, abs=0.001)),
        ('phosphenes-59', approx(0.65513, abs=0.001)),
        ('phosphenes-77', approx(0.71916, abs=0.001)),
        ('phosphenes-100', approx(0.74847, abs=0.001)),
        ('segments-12', approx(0.66263, abs=0.001)),
        ('segments-16', approx(0.61250, abs=0.001)),
        ('segments-21', approx(0.63777, abs=0.001)),
        ('segments-27', approx(0.62361, abs=0.001)),
        ('segments-35', approx(0.63180, abs=0.001)),
        ('segments-46', approx(0.65888, abs=0.001)),
        ('segments-59', approx(0.71097, abs=0.001)),
        ('segments-77', approx(0.77055, abs=0.001)),
        ('segments-100', approx(0.86305, abs=0.001)),  # all of the above are AccuracyDistance
        ('phosphenes-composite', approx(0.45755, abs=0.01)),  # composites are ErrorConsistency
        ('segments-composite', approx(0.42529, abs=0.01)),
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        if 'composite' in dataset:
            benchmark = f"Scialom2024_{dataset}BehavioralErrorConsistency"
        else:
            benchmark = f"Scialom2024_{dataset}BehavioralAccuracyDistance"
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark.ceiling
        assert ceiling == expected_ceiling

    @pytest.mark.parametrize('dataset, expected_raw_score', [
        ('rgb', approx(0.99000, abs=0.001)),
        ('contours', approx(0.52458, abs=0.001)),
        ('phosphenes-12', approx(0.73166, abs=0.001)),
        ('phosphenes-16', approx(0.70666, abs=0.001)),
        ('phosphenes-21', approx(0.69333, abs=0.001)),
        ('phosphenes-27', approx(0.61833, abs=0.001)),
        ('phosphenes-35', approx(0.56583, abs=0.001)),
        ('phosphenes-46', approx(0.47500, abs=0.001)),
        ('phosphenes-59', approx(0.40000, abs=0.001)),
        ('phosphenes-77', approx(0.32416, abs=0.001)),
        ('phosphenes-100', approx(0.24916, abs=0.001)),
        ('segments-12', approx(0.69500, abs=0.001)),
        ('segments-16', approx(0.63250, abs=0.001)),
        ('segments-21', approx(0.58666, abs=0.001)),
        ('segments-27', approx(0.51833, abs=0.001)),
        ('segments-35', approx(0.45333, abs=0.001)),
        ('segments-46', approx(0.38916, abs=0.001)),
        ('segments-59', approx(0.34083, abs=0.001)),
        ('segments-77', approx(0.25083, abs=0.001)),
        ('segments-100', approx(0.20750, abs=0.001)),  # all of the above are AccuracyDistance
        ('phosphenes-composite', approx(0.18057, abs=0.01)),  # composites are ErrorConsistency
        ('segments-composite', approx(0.15181, abs=0.01)),
    ])
    def test_model_8_degrees(self, dataset, expected_raw_score):
        if 'composite' in dataset:
            benchmark = f"Scialom2024_{dataset}BehavioralErrorConsistency"
        else:
            benchmark = f"Scialom2024_{dataset}BehavioralAccuracyDistance"
        benchmark = load_benchmark(benchmark)
        filename = f"resnet50_julios_Scialom2024_{dataset}.nc"
        precomputed_features = Path(__file__).parent / filename
        s3.download_file_if_not_exists(precomputed_features,
                                       bucket='brain-score-tests', remote_filepath=f'tests/test_benchmarks/{filename}')
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
                                       bucket='brain-score-tests', remote_filepath=f'tests/test_benchmarks/{filename}')
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        score = benchmark(precomputed_features)
        raw_score = score.raw
        # division by ceiling <= 1 should result in higher score
        assert score >= raw_score
        assert raw_score == expected_accuracy
