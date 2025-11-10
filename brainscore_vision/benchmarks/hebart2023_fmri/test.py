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
        ('Hebart2023_fmri.V1-ridgecv', approx(0.5409453749399302, abs=0.001)),
        ('Hebart2023_fmri.V2-ridgecv', approx(0.4188572857531148, abs=0.001)),
        ('Hebart2023_fmri.V4-ridgecv', approx(0.1753696707620169, abs=0.001)),
        ('Hebart2023_fmri.IT-ridgecv', approx(0.304639059406884, abs=0.001)),
    ])
    def test_model_score(self, benchmark, expected_score):
        benchmark = load_benchmark(benchmark)
        model = load_model('alexnet')
        score = benchmark(model)
        assert score == expected_score
