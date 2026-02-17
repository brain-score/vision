import pytest
from pytest import approx

from brainscore_vision import load_benchmark, load_model


# --- Registry tests: verify all 16 surface benchmark identifiers load correctly ---

@pytest.mark.private_access
class TestAllen2022fmriSurfaceBenchmarkRegistry:
    @pytest.mark.parametrize("region", ["V1", "V2", "V4", "IT"])
    def test_ridge_8subj(self, region):
        benchmark = load_benchmark(f"Allen2022_fmri_surface.{region}-ridge")
        assert benchmark.identifier == f"Allen2022_fmri_surface.{region}-ridge"
        assert benchmark.region == region
        assert benchmark._visual_degrees == 8.4

    @pytest.mark.parametrize("region", ["V1", "V2", "V4", "IT"])
    def test_rdm_8subj(self, region):
        benchmark = load_benchmark(f"Allen2022_fmri_surface.{region}-rdm")
        assert benchmark.identifier == f"Allen2022_fmri_surface.{region}-rdm"
        assert benchmark.region == region
        assert benchmark._visual_degrees == 8.4

    @pytest.mark.parametrize("region", ["V1", "V2", "V4", "IT"])
    def test_ridge_4subj(self, region):
        benchmark = load_benchmark(f"Allen2022_fmri_surface_4subj.{region}-ridge")
        assert benchmark.identifier == f"Allen2022_fmri_surface_4subj.{region}-ridge"
        assert benchmark.region == region
        assert benchmark._visual_degrees == 8.4

    @pytest.mark.parametrize("region", ["V1", "V2", "V4", "IT"])
    def test_rdm_4subj(self, region):
        benchmark = load_benchmark(f"Allen2022_fmri_surface_4subj.{region}-rdm")
        assert benchmark.identifier == f"Allen2022_fmri_surface_4subj.{region}-rdm"
        assert benchmark.region == region
        assert benchmark._visual_degrees == 8.4


# --- AlexNet scoring tests: verify ceiling-normalized scores match expected values ---

@pytest.mark.private_access
@pytest.mark.slow
class TestAllen2022fmriSurfaceAlexNetRidge:
    @pytest.fixture(scope="class")
    def model(self):
        return load_model('alexnet')

    @pytest.mark.parametrize('benchmark_id, expected_score', [
        ('Allen2022_fmri_surface.V1-ridge', approx(0.4129, abs=0.005)),
        ('Allen2022_fmri_surface.V2-ridge', approx(0.4409, abs=0.005)),
        ('Allen2022_fmri_surface.V4-ridge', approx(0.4009, abs=0.005)),
        ('Allen2022_fmri_surface.IT-ridge', approx(0.2975, abs=0.005)),
    ])
    def test_8subj(self, model, benchmark_id, expected_score):
        benchmark = load_benchmark(benchmark_id)
        score = benchmark(model)
        assert float(score) == expected_score

    @pytest.mark.parametrize('benchmark_id, expected_score', [
        ('Allen2022_fmri_surface_4subj.V1-ridge', approx(0.4062, abs=0.005)),
        ('Allen2022_fmri_surface_4subj.V2-ridge', approx(0.4303, abs=0.005)),
        ('Allen2022_fmri_surface_4subj.V4-ridge', approx(0.3690, abs=0.005)),
        ('Allen2022_fmri_surface_4subj.IT-ridge', approx(0.2804, abs=0.005)),
    ])
    def test_4subj(self, model, benchmark_id, expected_score):
        benchmark = load_benchmark(benchmark_id)
        score = benchmark(model)
        assert float(score) == expected_score


@pytest.mark.private_access
@pytest.mark.slow
class TestAllen2022fmriSurfaceAlexNetRDM:
    @pytest.fixture(scope="class")
    def model(self):
        return load_model('alexnet')

    @pytest.mark.parametrize('benchmark_id, expected_score', [
        ('Allen2022_fmri_surface.V4-rdm', approx(0.3713, abs=0.005)),
    ])
    def test_rdm_8subj(self, model, benchmark_id, expected_score):
        benchmark = load_benchmark(benchmark_id)
        score = benchmark(model)
        assert float(score) == expected_score

    @pytest.mark.parametrize('benchmark_id, expected_score', [
        ('Allen2022_fmri_surface_4subj.V4-rdm', approx(0.4768, abs=0.005)),
    ])
    def test_rdm_4subj(self, model, benchmark_id, expected_score):
        benchmark = load_benchmark(benchmark_id)
        score = benchmark(model)
        assert float(score) == expected_score
