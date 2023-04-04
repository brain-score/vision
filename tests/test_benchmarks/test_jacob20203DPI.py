from pathlib import Path
import pytest
from pytest import approx
from brainio.assemblies import BehavioralAssembly
from brainscore import benchmark_pool
from tests.test_benchmarks import PrecomputedFeatures


class TestJacob20203DPI:

    @pytest.mark.parametrize('benchmark', [
        'Jacob2020-3dpi_square',
        'Jacob2020-3dpi_y',
    ])
    def test_in_pool(self, benchmark):
        assert benchmark in benchmark_pool

    @pytest.mark.parametrize('benchmark, expected_ceiling', [
        ('Jacob2020-3dpi_square', approx(0.9228, abs=0.0001)),
        ('Jacob2020-3dpi_y', approx(0.9312, abs=0.001)),
    ])
    def test_benchmark_ceiling(self, benchmark, expected_ceiling):
        benchmark = benchmark_pool[benchmark]
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == expected_ceiling

    @pytest.mark.parametrize('benchmark, model, expected_raw_score', [
        ('Jacob2020-3dpi_square', 'alexnet', approx(0.5908, abs=0.0001)),
        # ('Jacob2020-3dpi_y', 'alexnet', approx(0.6344, abs=0.0001)),
        # ('Jacob2020-3dpi_square', 'resnet-18', approx(0.2460, abs=0.0001)),
        # ('Jacob2020-3dpi_y', 'resnet-18', approx(0.3695, abs=0.0001)),
    ])
    def test_model_raw_score(self, benchmark, model, expected_raw_score):
        # load features
        shape = benchmark.split("_")[-1]
        precomputed_features = Path(__file__).parent / \
                               f"model_identifier={model},benchmark_identifier={shape}-IT,visual_degrees=8.nc"
        benchmark = benchmark_pool[benchmark]
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=8.0,  # doesn't matter, features are already computed
                                                   )
        score = benchmark(precomputed_features)
        raw_score = score.raw

        # division by ceiling <= 1 should result in higher score
        assert score.sel(aggregation='center') >= raw_score.sel(aggregation='center')
        assert raw_score.sel(aggregation='center') == expected_raw_score

    @pytest.mark.parametrize('benchmark, model, expected_ceiled_score', [
        ('Jacob2020-3dpi_square', 'alexnet', approx(0.3000, abs=0.0001)),
        ('Jacob2020-3dpi_y', 'alexnet', approx(0.6344, abs=0.0001)),
        ('Jacob2020-3dpi_square', 'resnet-18', approx(0.2667, abs=0.0001)),
        ('Jacob2020-3dpi_y', 'resnet-18', approx(0.3968, abs=0.0001)),
    ])
    def test_model_ceiled_score(self, benchmark, model, expected_ceiled_score):
        shape = benchmark.split("_")[-1]
        precomputed_features = Path(__file__).parent / \
                               f"model_identifier={model},benchmark_identifier={shape}-IT,visual_degrees=8.nc"
        benchmark = benchmark_pool[benchmark]
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=8.0,
                                                   # doesn't matter, features are already computed
                                                   )
        score = benchmark(precomputed_features)
        assert score[0] == expected_ceiled_score
