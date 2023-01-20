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
        ('Baker2022-accuracy_delta_frankenstein', 'resnet-50-pytorch', approx(0.21135, abs=0.001)),
        # ('Baker2022-accuracy_delta_fragmented', approx(0.22546, abs=0.001)),
        # ('Baker2022-inverted_accuracy_delta', 'resnet-50-pytorch', approx(0.06800, abs=0.001)),
        # ('Baker2022-accuracy_delta_frankenstein', 'resnet50-SIN', approx(0.03626, abs=0.001)),
        # ('Baker2022-accuracy_delta_fragmented', 'resnet50-SIN', approx(0.21029, abs=0.001)),
        # ('Baker2022-inverted_accuracy_delta', 'resnet50-SIN', approx(0.25090, abs=0.001)),

    ])
    def test_model_raw_score(self, benchmark, model, expected_raw_score):
        benchmark = benchmark_pool[benchmark]
        # load features
        # precomputed_features = Path(__file__).parent / f'{model}-3deg-Geirhos2021_{dataset}.nc'
        # precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        # precomputed_features = PrecomputedFeatures(precomputed_features,
        #                                            visual_degrees=8.8,  # doesn't matter, features are already computed
        #                                            )
        # # score
        # score = benchmark(precomputed_features)
        score = benchmark
        raw_score = score.raw
        # division by ceiling <= 1 should result in higher score
        assert score.sel(aggregation='center') >= raw_score.sel(aggregation='center')
        assert raw_score.sel(aggregation='center') == expected_raw_score

    @pytest.mark.parametrize('model, expected_raw_score', [
        ('resnet-50-pytorch-3deg', approx(0.20834, abs=0.001)),
        ('resnet-50-pytorch-8deg', approx(0.10256, abs=0.001)),
    ])
    def test_model_mean(self, model, expected_raw_score):
        scores = []
        for dataset in DATASETS:
            benchmark = benchmark_pool[f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency"]
            precomputed_features = Path(__file__).parent / f'{model}-Geirhos2021_{dataset}.nc'
            precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
            # these features were packaged with condition as int/float. Current xarray versions have trouble when
            # selecting for a float coordinate however, so we had to change the type to string.
            precomputed_features = cast_coordinate_type(precomputed_features, 'condition', newtype=str)
            precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
            score = benchmark(precomputed_features).raw
            scores.append(score.sel(aggregation='center'))
        mean_score = np.mean(scores)
        assert mean_score == expected_raw_score
