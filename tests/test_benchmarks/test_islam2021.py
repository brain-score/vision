from pathlib import Path
import pytest
from pytest import approx
from brainscore import benchmark_pool
from tests.test_benchmarks import PrecomputedFeatures
from brainio.assemblies import DataAssembly
import os
 
class TestEngineering2:

    @pytest.mark.parametrize('features',[('alexnet-islam2021-classifier.5.nc')])
    @pytest.mark.parametrize('factor, area, expected_value', [
        ('shape', 'v1', approx(0.18310547, abs=0.001)),
        ('texture', 'v1', approx(0.30834961, abs=0.001)),
        ('shape', 'v2', approx(0.18310547, abs=0.001)),
        ('texture', 'v2', approx(0.30834961, abs=0.001)),
        ('shape', 'v4', approx(0.18310547, abs=0.001)),
        ('texture', 'v4', approx(0.30834961, abs=0.001)),
        ('shape', 'it', approx(0.18310547, abs=0.001)),
        ('texture', 'it', approx(0.30834961, abs=0.001)),
    ])      
    def test_dimensionality(self, features, factor, area, expected_value):
        benchmark = benchmark_pool[f'neil.Islam2021-{factor}_{area}_dimensionality']
        precomputed_features = Path(__file__).parent / features
        precomputed_features = DataAssembly.from_files(file_path = precomputed_features)
        stimulus_id = list(map(lambda x: x.split("/")[-1][:-4], precomputed_features['stimulus_path'].values))
        precomputed_features['stimulus_path'] = stimulus_id
        precomputed_features = precomputed_features.rename({'stimulus_path': 'stimulus_id'})
        precomputed_features = precomputed_features.stack(presentation=('stimulus_id',))
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=None)
        score = benchmark(precomputed_features)
        assert score.item() == expected_value 
        
    def lstrip_local(self,path):
        parts = path.split(os.sep)
        start_index = parts.index('.brainio')
        path = os.sep.join(parts[start_index:])
        return path