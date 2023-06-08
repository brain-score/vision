from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore import benchmark_pool
from brainscore.benchmarks.malania2007 import DATASETS
from tests.test_benchmarks import PrecomputedFeatures


class TestBehavioral:
    def test_count(self):
        assert len(DATASETS) == 5 + 2 + 2

    @pytest.mark.parametrize('dataset', DATASETS)
    def test_in_pool(self, dataset):
        identifier = f"Malania2007_{dataset.replace('-', '')}"
        assert identifier in benchmark_pool

    def test_mean_ceiling(self):
        benchmarks = [f"Malania2007_{dataset.replace('-', '')}" for dataset in DATASETS]
        benchmarks = [benchmark_pool[benchmark] for benchmark in benchmarks]
        ceilings = [benchmark.ceiling.sel(aggregation='center') for benchmark in benchmarks]
        mean_ceiling = np.mean(ceilings)
        assert mean_ceiling == approx(0.7724487108297781, abs=0.001)  # TODO: check that this is correct

    # these test values are for the pooled score ceiling
    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('short-2', approx(0.82203635, abs=0.001)),
        ('short-4', approx(0.78841608, abs=0.001)),
        ('short-6', approx(0.80555853, abs=0.001)),
        ('short-8', approx(0.7866628, abs=0.001)),
        ('short-16', approx(0.90941085, abs=0.001)),
        ('equal-2', approx(0.77990816, abs=0.001)),
        ('long-2', approx(0.72215817, abs=0.001)),
        ('equal-16', approx(0.62778544, abs=0.001)),
        ('long-16', approx(0.71010202, abs=0.001))
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        benchmark = f"Malania2007_{dataset.replace('-', '')}"
        benchmark = benchmark_pool[benchmark]
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center').values.item() == expected_ceiling

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
        benchmark = benchmark_pool[f"Malania_{dataset.replace('-', '')}"]
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-Malania2007_{dataset}.nc'
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=8,  # doesn't matter, features are already computed
                                                   )
        # score
        score = benchmark(precomputed_features).raw
        assert score == expected_raw_score

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
    def test_model_8degrees(self, dataset, model, expected_raw_score):
        benchmark = benchmark_pool[f"Malania_{dataset.replace('-', '')}"]
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-Malania2007_{dataset}.nc'
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=3,  # doesn't matter, features are already computed
                                                   )
        # score
        score = benchmark(precomputed_features).raw
        assert score == expected_raw_score
