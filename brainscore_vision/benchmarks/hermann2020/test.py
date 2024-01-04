from pathlib import Path

import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.data_helpers import s3


class TestEngineering:
    @pytest.mark.parametrize('model, benchmark, expected_shape_bias', [
        ('resnet-50-pytorch', "Hermann2020cueconflict-shape_bias", approx(0.21392405, abs=0.001)),
        ('resnet-50-pytorch', "Hermann2020cueconflict-shape_match", approx(0.14083333, abs=0.001)),
    ])
    def test_score(self, model, benchmark, expected_shape_bias):
        benchmark = load_benchmark(benchmark)
        # load features
        filename = f'{model}-3deg-Geirhos2021_cue-conflict.nc'
        filepath = Path(__file__).parent / filename
        s3.download_file_if_not_exists(local_path=filepath,
                                       bucket='brain-score-tests', remote_filepath=f'tests/test_benchmarks/{filename}')
        precomputed_features = BehavioralAssembly.from_files(file_path=filepath)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=None)
        # score
        score = benchmark(precomputed_features)
        assert score == expected_shape_bias
