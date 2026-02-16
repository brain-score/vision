import pytest

from brainscore_vision import load_benchmark


@pytest.mark.private_access
class TestAllen2022fmriSurfaceBenchmarks:
    @pytest.mark.parametrize("region", ["V1", "V2", "V4", "IT"])
    def test_benchmark_registry(self, region):
        benchmark = load_benchmark(f"Allen2022_fmri_surface.{region}-ridge")
        assert benchmark.identifier == f"Allen2022_fmri_surface.{region}-ridge"
        assert benchmark.region == region
        assert benchmark._visual_degrees == 8.4


@pytest.mark.private_access
class TestAllen2022fmriSurfaceRSABenchmarks:
    @pytest.mark.parametrize("region", ["V1", "V2", "V4", "IT"])
    def test_benchmark_registry(self, region):
        benchmark = load_benchmark(f"Allen2022_fmri_surface.{region}-rdm")
        assert benchmark.identifier == f"Allen2022_fmri_surface.{region}-rdm"
        assert benchmark.region == region
        assert benchmark._visual_degrees == 8.4
