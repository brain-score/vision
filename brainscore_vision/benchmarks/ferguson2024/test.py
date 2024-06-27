from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.data_helpers import s3


class TestExist:
    @pytest.mark.parametrize('benchmark', [
        'Ferguson2024circle_line-value_delta',
        'Ferguson2024color-value_delta',
        'Ferguson2024convergence-value_delta',
        'Ferguson2024eighth-value_delta',
        'Ferguson2024gray_easy-value_delta',
        'Ferguson2024gray_hard-value_delta',
        'Ferguson2024half-value_delta',
        'Ferguson2024juncture-value_delta',
        'Ferguson2024lle-value_delta',
        'Ferguson2024llh-value_delta',
        'Ferguson2024quarter-value_delta',
        'Ferguson2024round_f-value_delta',
        'Ferguson2024round_v-value_delta',
        'Ferguson2024tilted_line-value_delta',
    ])
    def test_benchmark_registry(self, benchmark):
        assert benchmark in benchmark_registry


class TestBehavioral:
    @pytest.mark.private_access
    @pytest.mark.parametrize('benchmark, expected_ceiling', [
        ('Ferguson2024circle_line-value_delta', approx(0.883, abs=0.001)),
        ('Ferguson2024color-value_delta', approx(0.897, abs=0.001)),
        ('Ferguson2024convergence-value_delta', approx(0.862, abs=0.001)),
        ('Ferguson2024eighth-value_delta', approx(0.852, abs=0.001)),
        ('Ferguson2024gray_easy-value_delta', approx(0.907, abs=0.001)),
        ('Ferguson2024gray_hard-value_delta', approx(0.863, abs=0.001)),
        ('Ferguson2024half-value_delta', approx(0.898, abs=0.001)),
        ('Ferguson2024juncture-value_delta', approx(0.767, abs=0.001)),
        ('Ferguson2024lle-value_delta', approx(0.831, abs=0.001)),
        ('Ferguson2024llh-value_delta', approx(0.812, abs=0.001)),
        ('Ferguson2024quarter-value_delta', approx(0.876, abs=0.001)),
        ('Ferguson2024round_f-value_delta', approx(0.874, abs=0.001)),
        ('Ferguson2024round_v-value_delta', approx(0.853, abs=0.001)),
        ('Ferguson2024tilted_line-value_delta', approx(0.912, abs=0.001)),
    ])
    def test_benchmark_ceiling(self, benchmark, expected_ceiling):
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark._ceiling
        assert ceiling == expected_ceiling

    @pytest.mark.private_access
    @pytest.mark.parametrize('benchmark, expected_raw_score', [
        ('Ferguson2024circle_line-value_delta', approx(0.883, abs=0.001)),
        # ('Ferguson2024color-value_delta', approx(0.897, abs=0.001)),
        # ('Ferguson2024convergence-value_delta', approx(0.862, abs=0.001)),
        # ('Ferguson2024eighth-value_delta', approx(0.852, abs=0.001)),
        # ('Ferguson2024gray_easy-value_delta', approx(0.907, abs=0.001)),
        # ('Ferguson2024gray_hard-value_delta', approx(0.863, abs=0.001)),
        # ('Ferguson2024half-value_delta', approx(0.898, abs=0.001)),
        # ('Ferguson2024juncture-value_delta', approx(0.767, abs=0.001)),
        # ('Ferguson2024lle-value_delta', approx(0.831, abs=0.001)),
        # ('Ferguson2024llh-value_delta', approx(0.812, abs=0.001)),
        # ('Ferguson2024quarter-value_delta', approx(0.876, abs=0.001)),
        # ('Ferguson2024round_f-value_delta', approx(0.874, abs=0.001)),
        # ('Ferguson2024round_v-value_delta', approx(0.853, abs=0.001)),
        # ('Ferguson2024tilted_line-value_delta', approx(0.912, abs=0.001)),
    ])
    def test_model_raw_score(self, benchmark, expected_raw_score):
        benchmark = load_benchmark(benchmark)
        # filename = f"alexnet_{benchmark}.nc"
        # precomputed_features = Path(__file__).parent / filename
        # s3.download_file_if_not_exists(precomputed_features,
        #                                bucket='brainscore-vision', remote_filepath=f'benchmarks/Ferguson2024/{filename}')

        precomputed_features = BehavioralAssembly.from_files(file_path="alexnet_Ferguson2024circle_line-value_delta_updated.nc")
        #precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        score = benchmark(precomputed_features)
        raw_score = score.raw
        # division by ceiling <= 1 should result in higher score
        assert score >= raw_score
        assert raw_score == expected_raw_score

    @pytest.mark.private_access
    @pytest.mark.parametrize('benchmark, expected_raw_score', [
        ('Ferguson2024circle_line-value_delta', approx(0.883, abs=0.001)),
        ('Ferguson2024color-value_delta', approx(0.897, abs=0.001)),
        ('Ferguson2024convergence-value_delta', approx(0.862, abs=0.001)),
        ('Ferguson2024eighth-value_delta', approx(0.852, abs=0.001)),
        ('Ferguson2024gray_easy-value_delta', approx(0.907, abs=0.001)),
        ('Ferguson2024gray_hard-value_delta', approx(0.863, abs=0.001)),
        ('Ferguson2024half-value_delta', approx(0.898, abs=0.001)),
        ('Ferguson2024juncture-value_delta', approx(0.767, abs=0.001)),
        ('Ferguson2024lle-value_delta', approx(0.831, abs=0.001)),
        ('Ferguson2024llh-value_delta', approx(0.812, abs=0.001)),
        ('Ferguson2024quarter-value_delta', approx(0.876, abs=0.001)),
        ('Ferguson2024round_f-value_delta', approx(0.874, abs=0.001)),
        ('Ferguson2024round_v-value_delta', approx(0.853, abs=0.001)),
        ('Ferguson2024tilted_line-value_delta', approx(0.912, abs=0.001)),
    ])
    def test_model_ceiled_score(self, benchmark, expected_raw_score):
        benchmark = load_benchmark(benchmark)
        filename = f"alexnet_{benchmark}.nc"

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

