import pytest
from pytest import approx

from brainscore_vision import benchmark_registry, load_benchmark, load_model


class TestExist:
    @pytest.mark.private_access
    @pytest.mark.parametrize("identifier", [
        'Hebart2023_fmri.V1-ridge',
        'Hebart2023_fmri.V2-ridge',
        'Hebart2023_fmri.V4-ridge',
        'Hebart2023_fmri.IT-ridge',
        'Hebart2023_fmri.V1-ridgecv',
        'Hebart2023_fmri.V2-ridgecv',
        'Hebart2023_fmri.V4-ridgecv',
        'Hebart2023_fmri.IT-ridgecv',
    ])
    def test_benchmark_registry(self, identifier):
        benchmark = load_benchmark(identifier)
        assert benchmark is not None
        assert benchmark.identifier == identifier


class TestAlexNet:
    @pytest.mark.private_access
    @pytest.mark.slow
    @pytest.mark.parametrize('benchmark, expected_score', [
        ('Hebart2023_fmri.V1-ridgecv', approx(0.5489315250178326, abs=0.001)),
        ('Hebart2023_fmri.V2-ridgecv', approx(0.38434124185586294, abs=0.001)),
        ('Hebart2023_fmri.V4-ridgecv', approx(0.18973093479440584, abs=0.001)),
        ('Hebart2023_fmri.IT-ridgecv', approx(0.3103578965860336, abs=0.001)),
    ])
    def test_model_score(self, benchmark, expected_score):
        benchmark = load_benchmark(benchmark)
        model = load_model('alexnet')
        score = benchmark(model)
        assert score == expected_score
