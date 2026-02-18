from pytest import approx
import pytest

from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.benchmark_helpers.test_helper import (
    StandardizedTests,
    NumberOfTrialsTests,
)
from brainscore_vision.benchmarks.muzellec2026.benchmark import (
    MajajHongV4PublicBenchmark,
    MajajHongITPublicBenchmark,
)

standardized_tests = StandardizedTests()
num_trials_test = NumberOfTrialsTests()


@pytest.mark.parametrize("benchmark_identifier", [
    "MajajHong2015public.V4-reverse_pls",
    "MajajHong2015public.IT-reverse_pls",
])
def test_benchmark_registry(benchmark_identifier):
    assert benchmark_identifier in benchmark_registry


@pytest.mark.parametrize("benchmark_ctor", [MajajHongV4PublicBenchmark, MajajHongITPublicBenchmark])
def test_constructs(benchmark_ctor):
    b = benchmark_ctor()
    assert b is not None
    assert hasattr(b, "_assembly")
    assert b._assembly is not None


@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("benchmark_identifier, expected", [
    pytest.param("MajajHong2015public.V4-reverse_pls", approx(0.8837782145, abs=0.005)),
    pytest.param("MajajHong2015public.IT-reverse_pls", approx(0.8157994151, abs=0.005)),
])
def test_ceilings(benchmark_identifier, expected):
    standardized_tests.ceilings_test(benchmark_identifier, expected)


@pytest.mark.private_access
@pytest.mark.parametrize("benchmark_identifier", [
    "MajajHong2015public.V4-reverse_pls",
    "MajajHong2015public.IT-reverse_pls",
])
def test_repetitions(benchmark_identifier):
    num_trials_test.repetitions_test(benchmark_identifier)


@pytest.mark.memory_intense
@pytest.mark.parametrize("benchmark_ctor, visual_degrees, expected_raw, expected_ceiled", [
    pytest.param(
        MajajHongV4PublicBenchmark, 8,
        approx(0.8964847480, abs=0.01),
        approx(0.9093739699, abs=0.01),
    ),
    pytest.param(
        MajajHongITPublicBenchmark, 8,
        approx(0.8147792876, abs=0.01),
        approx(0.8137604357, abs=0.01),
    ),
])
def test_self(benchmark_ctor, visual_degrees, expected_raw, expected_ceiled):
    benchmark = benchmark_ctor()
    source = benchmark._assembly.copy()
    source = {benchmark._assembly.stimulus_set.identifier: source}
    score = benchmark(PrecomputedFeatures(source, visual_degrees=visual_degrees))

    assert float(score.raw) == expected_raw
    assert float(score) == expected_ceiled


@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("benchmark_identifier", [
    "MajajHong2015public.V4-reverse_pls",
    "MajajHong2015public.IT-reverse_pls",
])
def test_ceiling_has_error_attr(benchmark_identifier):
    b = load_benchmark(benchmark_identifier)
    ceiling = b.ceiling
    assert "error" in ceiling.attrs
