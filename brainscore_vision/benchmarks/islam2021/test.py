from pathlib import Path

import boto3
import pytest
from pytest import approx

from brainio.assemblies import DataAssembly
from brainscore_vision import load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures


@pytest.fixture()
def alexnet_features():
    # load
    filename = 'alexnet-islam2021-classifier.5.nc'
    precomputed_features_path = Path(__file__).parent / filename
    if not precomputed_features_path.is_file():  # download on demand
        s3 = boto3.client('s3')
        s3.download_file('brain-score-tests', f'tests/test_benchmarks/{filename}',
                         str(precomputed_features_path.absolute()))
    precomputed_features = DataAssembly.from_files(file_path=precomputed_features_path)

    # adjust metadata
    stimulus_id = list(map(lambda x: x.split("/")[-1][:-4], precomputed_features['stimulus_path'].values))
    precomputed_features['stimulus_path'] = stimulus_id
    precomputed_features = precomputed_features.rename({'stimulus_path': 'stimulus_id'})
    precomputed_features = precomputed_features.stack(presentation=('stimulus_id',))
    precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=None)
    return precomputed_features


@pytest.mark.private_access
class TestEngineering:
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
    def test_dimensionality(self, factor, area, expected_value, alexnet_features):
        benchmark = load_benchmark(f'Islam2021-{factor}_{area}_dimensionality')
        score = benchmark(alexnet_features)
        assert score.item() == expected_value
