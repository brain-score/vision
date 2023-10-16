from pathlib import Path
import pytest
from pytest import approx
from brainio.assemblies import BehavioralAssembly
from brainscore import benchmark_pool
from brainscore.benchmarks.hebart2023 import DATASETS
from tests.test_benchmarks import PrecomputedFeatures

"""
@pytest.mark.private_access
class TestHebart2022:

    # ensure dataset is there
    def test_count(self):
        assert len(DATASETS) == 1

    # ensure the benchmark itself is there
    @pytest.mark.parametrize('benchmark', [
        'Hebart2023'
    ])
    def test_in_pool(self, benchmark):
        assert benchmark in benchmark_pool

    # Test expected ceiling
    @pytest.mark.parametrize('benchmark, expected_ceiling', [
        ('Hebart2023', 0.6844), # TODO
    ])
    def test_benchmark_ceiling(self, benchmark, expected_ceiling):
        benchmark = benchmark_pool[benchmark]
        assembly = benchmark._assembly
        ceiling = benchmark._ceiling(assembly) # TODO might need some fixing
        assert ceiling == approx(expected_ceiling, abs=0.001)

    # Test raw scores
    @pytest.mark.parametrize('benchmark, model, expected_raw_score', [
        ('Hebart2023', 'resnet-50-pytorch', approx(None, abs=0.0001)), # TODO
        ('Hebart2023', 'resnet50-SIN', approx(None, abs=0.0001)), # TODO
    ])
    def test_model_raw_score(self, benchmark, model, expected_raw_score):
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-{benchmark}.nc'
        benchmark = benchmark_pool[benchmark]
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=6, 
                                                   )
        score = benchmark(precomputed_features)
        raw_score = score.raw
        assert raw_score.sel(aggregation='center') == expected_raw_score

        # division by ceiling <= 1 should result in higher score
        assert score.sel(aggregation='center') >= raw_score.sel(aggregation='center')

    # test ceiled score
    @pytest.mark.parametrize('benchmark, model, expected_ceiled_score', [
        ('Hebart2023', 'resnet-50-pytorch', approx(None, abs=0.0001)), # TODO
        ('Hebart2023', 'resnet50-SIN', approx(None, abs=0.0001)), # TODO
    ])
    def test_model_ceiled_score(self, benchmark, model, expected_ceiled_score):
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-{benchmark}.nc'
        benchmark = benchmark_pool[benchmark]
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=6,  
                                                   )
        score = benchmark(precomputed_features)
        assert score.sel(aggregation='center') == expected_ceiled_score

"""