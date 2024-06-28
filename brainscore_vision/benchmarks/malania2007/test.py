from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.benchmarks.malania2007.benchmark import DATASETS


class TestBehavioral:
    def test_count(self):
        assert len(DATASETS) == 5 + 2 + 2 + 1

    @pytest.mark.parametrize('dataset', DATASETS)
    def test_in_pool(self, dataset):
        identifier = f"Malania2007_{dataset}"
        assert identifier in benchmark_registry

    @pytest.mark.private_access
    # TODO: recompute
    def test_mean_ceiling(self):
        benchmarks = [f"Malania2007_{dataset}" for dataset in DATASETS]
        benchmarks = [load_benchmark(benchmark) for benchmark in benchmarks]
        ceilings = [benchmark.ceiling for benchmark in benchmarks]
        mean_ceiling = np.mean(ceilings)
        assert mean_ceiling == approx(0.5757928329186803, abs=0.001)

    # these test values are for the pooled score ceiling
    @pytest.mark.private_access
    # TODO: ceiling for vernier acuity
    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('short-2', approx(0.78719345, abs=0.001)),
        ('short-4', approx(0.49998989, abs=0.001)),
        ('short-6', approx(0.50590051, abs=0.001)),
        ('short-8', approx(0.4426336, abs=0.001)),
        ('short-16', approx(0.8383443, abs=0.001)),
        ('equal-2', approx(0.56664015, abs=0.001)),
        ('long-2', approx(0.46470421, abs=0.001)),
        ('equal-16', approx(0.44087153, abs=0.001)),
        ('long-16', approx(0.50996587, abs=0.001)),
        ('vernieracuity', approx(0.70168481, abs=0.001))
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        benchmark = f"Malania2007_{dataset}"
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center').values.item() == expected_ceiling

    @pytest.mark.private_access
    @pytest.mark.parametrize('dataset, model, expected_raw_score', [
        ('short-2', 'resnet-18', approx(0., abs=0.001)),
        ('short-4', 'resnet-18', approx(0., abs=0.001)),
        ('short-6', 'resnet-18', approx(0., abs=0.001)),
        ('short-8', 'resnet-18', approx(0., abs=0.001)),
        ('short-16', 'resnet-18', approx(0., abs=0.001)),
        ('equal-2', 'resnet-18', approx(0., abs=0.001)),
        ('long-2', 'resnet-18', approx(0., abs=0.001)),
        ('equal-16', 'resnet-18', approx(0., abs=0.001)),
        ('long-16', 'resnet-18', approx(0., abs=0.001)),
    ])
    def test_model_8degrees(self, dataset, model, expected_raw_score):
        raise Exception("This test needs to be recalculated.")
        benchmark = benchmark_registry[f"Malania2007_{dataset}"]
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-Malania2007_{dataset}.nc'
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=8,  # doesn't matter, features are already computed
                                                   )
        # score
        score = benchmark(precomputed_features).raw
        assert score == expected_raw_score

    @pytest.mark.private_access
    @pytest.mark.parametrize('dataset, model, expected_raw_score', [
        ('short-2', 'resnet-18-3deg', approx(0., abs=0.001)),
        ('short-4', 'resnet-18-3deg', approx(0., abs=0.001)),
        ('short-6', 'resnet-18-3deg', approx(0., abs=0.001)),
        ('short-8', 'resnet-18-3deg', approx(0., abs=0.001)),
        ('short-16', 'resnet-18-3deg', approx(0., abs=0.001)),
        ('equal-2', 'resnet-18-3deg', approx(0., abs=0.001)),
        ('long-2', 'resnet-18-3deg', approx(0., abs=0.001)),
        ('equal-16', 'resnet-18-3deg', approx(0., abs=0.001)),
        ('long-16', 'resnet-18-3deg', approx(0., abs=0.001)),
    ])
    def test_model_3degrees(self, dataset, model, expected_raw_score):
        raise Exception("This test needs to be recalculated.")
        benchmark = benchmark_registry[f"Malania2007_{dataset}"]
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-Malania2007_{dataset}.nc'
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=3,  # doesn't matter, features are already computed
                                                   )
        # score
        score = benchmark(precomputed_features).raw
        assert score == expected_raw_score
