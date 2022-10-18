from pathlib import Path
import pytest
from pytest import approx
from brainio.assemblies import BehavioralAssembly
from brainscore import benchmark_pool
from tests.test_benchmarks import PrecomputedFeatures


class TestEngineering:   
    @pytest.mark.parametrize('model, expected_shape_bias', [
        ('resnet-50-pytorch', approx(0.21392405, abs=0.001)),
    ])
    def test_shape_bias(self, model, expected_shape_bias):
        benchmark = benchmark_pool["kornblith.Hermann2020cueconflict-shape_bias"]
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-3deg-Geirhos2021_cue-conflict.nc'
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=None)
        # score
        score = benchmark(precomputed_features)
        assert score.sel(aggregation='center') == expected_shape_bias
        
        
    @pytest.mark.parametrize('model, expected_shape_match', [
        ('resnet-50-pytorch', approx(0.14083333, abs=0.001)),
    ])
    def test_shape_match(self, model, expected_shape_match):
        benchmark = benchmark_pool["kornblith.Hermann2020cueconflict-shape_match"]
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-3deg-Geirhos2021_cue-conflict.nc'
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=None)
        # score
        score = benchmark(precomputed_features)
        assert score.sel(aggregation='center') == expected_shape_match
        
     
  