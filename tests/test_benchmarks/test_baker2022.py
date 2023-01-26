from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore import benchmark_pool
from brainscore.benchmarks.baker2022 import DATASETS
from tests.test_benchmarks import PrecomputedFeatures


class TestBaker2022:
    def test_count(self):
        assert len(DATASETS) == 2

    @pytest.mark.parametrize('benchmark', [
        'Baker2022-inverted_accuracy_delta',
        'Baker2022-accuracy_delta_fragmented',
        'Baker2022-inverted_accuracy_delta'
    ])
    def test_in_pool(self, benchmark):
        assert benchmark in benchmark_pool

    @pytest.mark.parametrize('benchmark, expected_ceiling', [
        ('Baker2022-accuracy_delta_frankenstein', 0.8498),
        ('Baker2022-accuracy_delta_fragmented', 0.9385),
        ('Baker2022-inverted_accuracy_delta', 0.6538),
    ])
    def test_benchmark_ceiling(self, benchmark, expected_ceiling):
        benchmark = benchmark_pool[benchmark]
        assembly = benchmark._assembly
        if "inverted" in benchmark.identifier:
            inverted_assembly = assembly[assembly["orientation"] == "inverted"]
            ceiling = benchmark._ceiling(inverted_assembly)
        else:
            ceiling = benchmark._ceiling(assembly)
        assert ceiling == approx(expected_ceiling, abs=0.001)

    @pytest.mark.parametrize('benchmark, model, expected_raw_score', [
        ('Baker2022-accuracy_delta_frankenstein', 'resnet-50-pytorch', approx(0.2847, abs=0.0001)),
        ('Baker2022-accuracy_delta_fragmented', 'resnet-50-pytorch', approx(0.8452, abs=0.0001)),
        ('Baker2022-inverted_accuracy_delta', 'resnet-50-pytorch', approx(0.0, abs=0.0001)),
        ('Baker2022-accuracy_delta_frankenstein', 'resnet50-SIN', approx(0.6823, abs=0.0001)),
        ('Baker2022-accuracy_delta_fragmented', 'resnet50-SIN', approx(0.9100, abs=0.0001)),
        ('Baker2022-inverted_accuracy_delta', 'resnet50-SIN', approx(0.7050, abs=0.0001)),

    ])
    def test_model_raw_score(self, benchmark, model, expected_raw_score):

        # load features
        precomputed_features = Path(__file__).parent / f'{model}-{benchmark}.nc'
        benchmark = benchmark_pool[benchmark]
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=8.8,  # doesn't matter, features are already computed
                                                   )
        score = benchmark(precomputed_features)
        raw_score = score.raw
        assert raw_score[0] == expected_raw_score

        # division by ceiling <= 1 should result in higher score
        assert score.sel(aggregation='center') >= raw_score.sel(aggregation='center')
        assert raw_score.sel(aggregation='center') == expected_raw_score

    @pytest.mark.parametrize('benchmark, model, expected_ceiled_score', [
        ('Baker2022-accuracy_delta_frankenstein', 'resnet-50-pytorch', approx(0.3350, abs=0.0001)),
        ('Baker2022-accuracy_delta_fragmented', 'resnet-50-pytorch', approx(0.9005, abs=0.0001)),
        ('Baker2022-inverted_accuracy_delta', 'resnet-50-pytorch', approx(0.0, abs=0.0001)),
        ('Baker2022-accuracy_delta_frankenstein', 'resnet50-SIN', approx(0.8029, abs=0.0001)),
        ('Baker2022-accuracy_delta_fragmented', 'resnet50-SIN', approx(0.9696, abs=0.0001)),
        ('Baker2022-inverted_accuracy_delta', 'resnet50-SIN', approx(1.000, abs=0.0001)),
    ])
    def test_model_ceiled_score(self, benchmark, model, expected_ceiled_score):
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-{benchmark}.nc'
        benchmark = benchmark_pool[benchmark]
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=8.8,  # doesn't matter, features are already computed
                                                   )
        score = benchmark(precomputed_features)
        assert score[0] == expected_ceiled_score
