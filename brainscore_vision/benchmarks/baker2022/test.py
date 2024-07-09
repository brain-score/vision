from pathlib import Path
import pytest
from pytest import approx
from brainio.assemblies import BehavioralAssembly
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmarks.baker2022 import DATASETS
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.data_helpers import s3


@pytest.mark.private_access
class TestBaker2022:

    # ensure normal and inverted datasets are there
    def test_count(self):
        assert len(DATASETS) == 2

    # ensure the three benchmarks themselves are there
    @pytest.mark.parametrize('identifier', [
        'Baker2022inverted-accuracy_delta',
        'Baker2022fragmented-accuracy_delta',
        'Baker2022frankenstein-accuracy_delta',
    ])
    def test_identifier(self, identifier):
        assert identifier in benchmark_registry
        benchmark = load_benchmark(identifier=identifier)
        assert benchmark.identifier == identifier

    # Test expected ceiling
    @pytest.mark.parametrize('benchmark, expected_ceiling', [
        ('Baker2022frankenstein-accuracy_delta', 0.8498),
        ('Baker2022fragmented-accuracy_delta', 0.9385),
        ('Baker2022inverted-accuracy_delta', 0.6538),
    ])
    def test_benchmark_ceiling(self, benchmark, expected_ceiling):
        benchmark = load_benchmark(benchmark)
        assembly = benchmark._assembly
        if "inverted" in benchmark.identifier:
            inverted_assembly = assembly[assembly["orientation"] == "inverted"]
            ceiling = benchmark._ceiling(inverted_assembly)
        else:
            ceiling = benchmark._ceiling(assembly)
        assert ceiling == approx(expected_ceiling, abs=0.001)

    # Test raw scores
    @pytest.mark.parametrize('benchmark, model, expected_raw_score', [
        ('Baker2022frankenstein-accuracy_delta', 'resnet-50-pytorch', approx(0.2847, abs=0.0001)),
        ('Baker2022fragmented-accuracy_delta', 'resnet-50-pytorch', approx(0.8452, abs=0.0001)),
        ('Baker2022inverted-accuracy_delta', 'resnet-50-pytorch', approx(0.0, abs=0.0001)),
        ('Baker2022frankenstein-accuracy_delta', 'resnet50-SIN', approx(0.6823, abs=0.0001)),
        ('Baker2022fragmented-accuracy_delta', 'resnet50-SIN', approx(0.9100, abs=0.0001)),
        ('Baker2022inverted-accuracy_delta', 'resnet50-SIN', approx(0.7050, abs=0.0001)),
    ])
    def test_model_raw_score(self, benchmark, model, expected_raw_score):

        benchmark_object = load_benchmark(benchmark)
        filename = f"{model}-{benchmark}.nc"
        precomputed_features = Path(__file__).parent / filename
        s3.download_file_if_not_exists(precomputed_features,
                                       bucket='brainscore-vision',
                                       remote_filepath=f'benchmarks/Baker2022/{filename}')
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        score = benchmark_object(precomputed_features)
        raw_score = score.raw

        # division by ceiling <= 1 should result in higher score
        assert score >= raw_score
        assert raw_score == expected_raw_score

    # test ceiled score
    @pytest.mark.parametrize('benchmark, model, expected_ceiled_score', [
        ('Baker2022frankenstein-accuracy_delta', 'resnet-50-pytorch', approx(0.3350, abs=0.0001)),
        ('Baker2022fragmented-accuracy_delta', 'resnet-50-pytorch', approx(0.9005, abs=0.0001)),
        ('Baker2022inverted-accuracy_delta', 'resnet-50-pytorch', approx(0.0, abs=0.0001)),
        ('Baker2022frankenstein-accuracy_delta', 'resnet50-SIN', approx(0.8029, abs=0.0001)),
        ('Baker2022fragmented-accuracy_delta', 'resnet50-SIN', approx(0.9696, abs=0.0001)),
        ('Baker2022inverted-accuracy_delta', 'resnet50-SIN', approx(1.000, abs=0.0001)),
    ])
    def test_model_ceiled_score(self, benchmark, model, expected_ceiled_score):
        benchmark_object = load_benchmark(benchmark)
        filename = f"{model}-{benchmark}.nc"
        precomputed_features = Path(__file__).parent / filename
        s3.download_file_if_not_exists(precomputed_features,
                                       bucket='brainscore-vision',
                                       remote_filepath=f'benchmarks/Baker2022/{filename}')
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        score = benchmark_object(precomputed_features)
        assert score == expected_ceiled_score
