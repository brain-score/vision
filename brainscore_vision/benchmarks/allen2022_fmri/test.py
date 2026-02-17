import pytest
from pytest import approx

from brainscore_vision import load_benchmark, load_model


# --- Registry tests: verify all 16 benchmark identifiers load correctly ---

@pytest.mark.private_access
class TestAllen2022fmriBenchmarkRegistry:
    @pytest.mark.parametrize("region", ["V1", "V2", "V4", "IT"])
    def test_ridge_8subj(self, region):
        benchmark = load_benchmark(f"Allen2022_fmri.{region}-ridge")
        assert benchmark.identifier == f"Allen2022_fmri.{region}-ridge"
        assert benchmark.region == region
        assert benchmark._visual_degrees == 8.4

    @pytest.mark.parametrize("region", ["V1", "V2", "V4", "IT"])
    def test_rdm_8subj(self, region):
        benchmark = load_benchmark(f"Allen2022_fmri.{region}-rdm")
        assert benchmark.identifier == f"Allen2022_fmri.{region}-rdm"
        assert benchmark.region == region
        assert benchmark._visual_degrees == 8.4

    @pytest.mark.parametrize("region", ["V1", "V2", "V4", "IT"])
    def test_ridge_4subj(self, region):
        benchmark = load_benchmark(f"Allen2022_fmri_4subj.{region}-ridge")
        assert benchmark.identifier == f"Allen2022_fmri_4subj.{region}-ridge"
        assert benchmark.region == region
        assert benchmark._visual_degrees == 8.4

    @pytest.mark.parametrize("region", ["V1", "V2", "V4", "IT"])
    def test_rdm_4subj(self, region):
        benchmark = load_benchmark(f"Allen2022_fmri_4subj.{region}-rdm")
        assert benchmark.identifier == f"Allen2022_fmri_4subj.{region}-rdm"
        assert benchmark.region == region
        assert benchmark._visual_degrees == 8.4


# --- AlexNet scoring tests: verify ceiling-normalized scores match expected values ---

@pytest.mark.private_access
@pytest.mark.slow
class TestAllen2022fmriAlexNetRidge:
    @pytest.fixture(scope="class")
    def model(self):
        return load_model('alexnet')

    @pytest.mark.parametrize('benchmark_id, expected_score', [
        ('Allen2022_fmri.V1-ridge', approx(0.4019, abs=0.005)),
        ('Allen2022_fmri.V2-ridge', approx(0.4366, abs=0.005)),
        ('Allen2022_fmri.V4-ridge', approx(0.3482, abs=0.005)),
        ('Allen2022_fmri.IT-ridge', approx(0.2870, abs=0.005)),
    ])
    def test_8subj(self, model, benchmark_id, expected_score):
        benchmark = load_benchmark(benchmark_id)
        score = benchmark(model)
        assert float(score) == expected_score

    @pytest.mark.parametrize('benchmark_id, expected_score', [
        ('Allen2022_fmri_4subj.V1-ridge', approx(0.3965, abs=0.005)),
        ('Allen2022_fmri_4subj.V2-ridge', approx(0.4080, abs=0.005)),
        ('Allen2022_fmri_4subj.V4-ridge', approx(0.3208, abs=0.005)),
        ('Allen2022_fmri_4subj.IT-ridge', approx(0.2599, abs=0.005)),
    ])
    def test_4subj(self, model, benchmark_id, expected_score):
        benchmark = load_benchmark(benchmark_id)
        score = benchmark(model)
        assert float(score) == expected_score


@pytest.mark.private_access
@pytest.mark.slow
class TestAllen2022fmriAlexNetRDM:
    @pytest.fixture(scope="class")
    def model(self):
        return load_model('alexnet')

    @pytest.mark.parametrize('benchmark_id, expected_score', [
        ('Allen2022_fmri.V4-rdm', approx(0.3281, abs=0.005)),
    ])
    def test_rdm_8subj(self, model, benchmark_id, expected_score):
        benchmark = load_benchmark(benchmark_id)
        score = benchmark(model)
        assert float(score) == expected_score

    @pytest.mark.parametrize('benchmark_id, expected_score', [
        ('Allen2022_fmri_4subj.V4-rdm', approx(0.4259, abs=0.005)),
    ])
    def test_rdm_4subj(self, model, benchmark_id, expected_score):
        benchmark = load_benchmark(benchmark_id)
        score = benchmark(model)
        assert float(score) == expected_score
