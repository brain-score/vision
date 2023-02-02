from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore import benchmark_pool
from brainscore.benchmarks.zhu2019 import DATASETS
from tests.test_benchmarks import PrecomputedFeatures


@pytest.mark.private_access
class TestZhu2019:
    def test_count(self):
        assert len(DATASETS) == 1

    @pytest.mark.parametrize('benchmark', [
        'Zhu2019-accuracy',
        'Zhu2019-accuracy-engineering',
    ])
    def test_in_pool(self, benchmark):
        assert benchmark in benchmark_pool

    @pytest.mark.parametrize('benchmark, expected_ceiling', [
        ('Zhu2019-accuracy', approx(0.8113, abs=0.0001)),
        ('Zhu2019-accuracy-engineering', approx(1.0000, abs=0.0001)),
    ])
    def test_benchmark_ceiling(self, benchmark, expected_ceiling):
        benchmark = benchmark_pool[benchmark]
        assembly = benchmark._assembly
        ceiling = benchmark._ceiling(assembly)
        if "engineering" in benchmark.identifier:
            assert ceiling[(ceiling['aggregation'] == 'center')] == expected_ceiling
        else:
            assert ceiling == expected_ceiling

    @pytest.mark.parametrize('benchmark, model, expected_raw_score', [
        ('Zhu2019-accuracy', 'alexnet', approx(0.1880, abs=0.0001)),
        ('Zhu2019-accuracy-engineering', 'alexnet', approx(0.470, abs=0.001)),
        ('Zhu2019-accuracy', 'resnet-18', approx(0.204, abs=0.0001)),
        ('Zhu2019-accuracy-engineering', 'resnet-18', approx(0.506, abs=0.001)),
    ])
    def test_model_raw_score(self, benchmark, model, expected_raw_score):

        # load features
        precomputed_features = Path(__file__).parent / f'{model}-{benchmark}.nc'
        benchmark = benchmark_pool[benchmark]
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=8.0,  # doesn't matter, features are already computed
                                                   )
        score = benchmark(precomputed_features)
        raw_score = score.raw
        assert raw_score[0] == expected_raw_score

        # division by ceiling <= 1 should result in higher score
        assert score.sel(aggregation='center') >= raw_score.sel(aggregation='center')
        assert raw_score.sel(aggregation='center') == expected_raw_score

    @pytest.mark.parametrize('benchmark, model, expected_ceiled_score', [
        ('Zhu2019-accuracy', 'alexnet', approx(0.232, abs=0.001)),
        ('Zhu2019-accuracy-engineering', 'alexnet', approx(0.470, abs=0.001)),
        ('Zhu2019-accuracy', 'resnet-18', approx(0.251, abs=0.001)),
        ('Zhu2019-accuracy-engineering', 'resnet-18', approx(0.506, abs=0.001)),
    ])
    def test_model_ceiled_score(self, benchmark, model, expected_ceiled_score):
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-{benchmark}.nc'
        benchmark = benchmark_pool[benchmark]
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=8.0,  # doesn't matter, features are already computed
                                                   )
        score = benchmark(precomputed_features)
        assert score[0] == expected_ceiled_score