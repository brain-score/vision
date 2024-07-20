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
    'Scialom2024_segments-allBehavioralErrorConsistency',
    'Scialom2024_phosphenes-allBehavioralAccuracyDistance',
    'Scialom2024_segments-allBehavioralAccuracyDistance'
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry


class TestBehavioral:
    @pytest.mark.private_access
    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('rgb', approx(0.98484, abs=0.001)),
        ('contours', approx(0.97794, abs=0.001)),
        ('phosphenes-12', approx(0.94213, abs=0.001)),
        ('phosphenes-16', approx(0.90114, abs=0.001)),
        ('phosphenes-21', approx(0.89134, abs=0.001)),
        ('phosphenes-27', approx(0.77732, abs=0.001)),
        ('phosphenes-35', approx(0.78466, abs=0.001)),
        ('phosphenes-46', approx(0.78246, abs=0.001)),
        ('phosphenes-59', approx(0.79745, abs=0.001)),
        ('phosphenes-77', approx(0.83475, abs=0.001)),
        ('phosphenes-100', approx(0.86073, abs=0.001)),
        ('segments-12', approx(0.86167, abs=0.001)),
        ('segments-16', approx(0.8202, abs=0.001)),
        ('segments-21', approx(0.8079, abs=0.001)),
        ('segments-27', approx(0.78282, abs=0.001)),
        ('segments-35', approx(0.77724, abs=0.001)),
        ('segments-46', approx(0.85391, abs=0.001)),
        ('segments-59', approx(0.83410, abs=0.001)),
        ('segments-77', approx(0.86227, abs=0.001)),
        ('segments-100', approx(0.92517, abs=0.001)),  # all of the above are AccuracyDistance
        ('phosphenes-allBehavioralErrorConsistency', approx(0.45755, abs=0.01)),
        ('segments-allBehavioralErrorConsistency', approx(0.42529, abs=0.01)),
        ('phosphenes-allBehavioralAccuracyDistance', approx(0.89533, abs=0.01)),
        ('segments-allBehavioralAccuracyDistance', approx(0.89052, abs=0.01)),
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        if 'all' in dataset:
            benchmark = f"Scialom2024_{dataset}"
        else:
            benchmark = f"Scialom2024_{dataset}BehavioralAccuracyDistance"
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark.ceiling
        assert ceiling == expected_ceiling

    @pytest.mark.private_access
    @pytest.mark.parametrize('dataset, expected_raw_score', [
        ('rgb', approx(0.92616, abs=0.001)),
        ('contours', approx(0.25445, abs=0.001)),
        ('phosphenes-12', approx(0.84177, abs=0.001)),
        ('phosphenes-16', approx(0.77513, abs=0.001)),
        ('phosphenes-21', approx(0.76437, abs=0.001)),
        ('phosphenes-27', approx(0.56895, abs=0.001)),
        ('phosphenes-35', approx(0.52008, abs=0.001)),
        ('phosphenes-46', approx(0.29478, abs=0.001)),
        ('phosphenes-59', approx(0.19022, abs=0.001)),
        ('phosphenes-77', approx(0.13569, abs=0.001)),
        ('phosphenes-100', approx(0.11234, abs=0.001)),
        ('segments-12', approx(0.72937, abs=0.001)),
        ('segments-16', approx(0.57043, abs=0.001)),
        ('segments-21', approx(0.49300, abs=0.001)),
        ('segments-27', approx(0.30014, abs=0.001)),
        ('segments-35', approx(0.22442, abs=0.001)),
        ('segments-46', approx(0.14312, abs=0.001)),
        ('segments-59', approx(0.12072, abs=0.001)),
        ('segments-77', approx(0.12996, abs=0.001)),
        ('segments-100', approx(0.11540, abs=0.001)),  # all of the above are AccuracyDistance
        ('phosphenes-all', approx(0.18057, abs=0.01)),  # alls are ErrorConsistency
        ('segments-all', approx(0.15181, abs=0.01)),
    ])
    def test_model_8_degrees(self, dataset, expected_raw_score):
        if 'all' in dataset:
            benchmark = f"Scialom2024_{dataset}BehavioralErrorConsistency"
        else:
            benchmark = f"Scialom2024_{dataset}BehavioralAccuracyDistance"
        benchmark = load_benchmark(benchmark)
        nc_filename = dataset.upper() if dataset == 'rgb' else dataset
        filename = f"resnet50_julios_Scialom2024_{nc_filename}.nc"
        precomputed_features = Path(__file__).parent / filename
        s3.download_file_if_not_exists(precomputed_features,
                                       bucket='brainscore-vision', remote_filepath=f'benchmarks/Scialom2024/{filename}')
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
        ('rgb', approx(0.91666, abs=0.001)),
        ('contours', approx(0.25000, abs=0.001)),
        ('phosphenes-12', approx(0.08333, abs=0.001)),
        ('phosphenes-16', approx(0.08333, abs=0.001)),
        ('phosphenes-21', approx(0.08333, abs=0.001)),
        ('phosphenes-27', approx(0.08333, abs=0.001)),
        ('phosphenes-35', approx(0.12500, abs=0.001)),
        ('phosphenes-46', approx(0.08333, abs=0.001)),
        ('phosphenes-59', approx(0.08333, abs=0.001)),
        ('phosphenes-77', approx(0.08333, abs=0.001)),
        ('phosphenes-100', approx(0.08333, abs=0.001)),
        ('segments-12', approx(0.08333, abs=0.001)),
        ('segments-16', approx(0.08333, abs=0.001)),
        ('segments-21', approx(0.08333, abs=0.001)),
        ('segments-27', approx(0.08333, abs=0.001)),
        ('segments-35', approx(0.08333, abs=0.001)),
        ('segments-46', approx(0.08333, abs=0.001)),
        ('segments-59', approx(0.08333, abs=0.001)),
        ('segments-77', approx(0.10416, abs=0.001)),
        ('segments-100', approx(0.10416, abs=0.001))
    ])
    def test_accuracy(self, dataset, expected_accuracy):
        benchmark = load_benchmark(f"Scialom2024_{dataset}EngineeringAccuracy")
        nc_filename = dataset.upper() if dataset == 'rgb' else dataset
        filename = f"resnet50_julios_Scialom2024_{nc_filename}.nc"
        precomputed_features = Path(__file__).parent / filename
        s3.download_file_if_not_exists(precomputed_features,
                                       bucket='brainio-brainscore', remote_filepath=f'/benchmarks/Scialom2024/{filename}')
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        score = benchmark(precomputed_features)
        raw_score = score.raw
        # division by ceiling <= 1 should result in higher score
        assert score >= raw_score
        assert raw_score == expected_accuracy
