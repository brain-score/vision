import pytest
from pytest import approx

from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.benchmark_helpers.test_helper import NumberOfTrialsTests

from brainscore_vision.benchmarks.muzellec2026.benchmark import (
    MajajHongV4PublicBenchmark,
    MajajHongITPublicBenchmark,
)

num_trials_test = NumberOfTrialsTests()

BENCHMARKS = [
    "MajajHong2015public.V4-reverse_pls",
    "MajajHong2015public.IT-reverse_pls",
]


@pytest.mark.parametrize("benchmark_identifier", BENCHMARKS)
def test_registered(benchmark_identifier):
    # Note: registry may not be populated until the benchmarks package is imported.
    assert benchmark_identifier in benchmark_registry


@pytest.mark.parametrize("benchmark_ctor", [MajajHongV4PublicBenchmark, MajajHongITPublicBenchmark])
def test_constructs(benchmark_ctor):
    b = benchmark_ctor()
    assert b is not None
    assert hasattr(b, "_assembly")
    assert b._assembly is not None


@pytest.mark.parametrize("benchmark_identifier", BENCHMARKS)
def test_repetitions_metadata(benchmark_identifier):
    # This checks that benchmark.number_of_trials etc are consistent with the assembly structure
    # and will fail if the benchmark lost repetition information needed for trial averaging.
    num_trials_test.repetitions_test(benchmark_identifier)


@pytest.mark.parametrize("benchmark_identifier", BENCHMARKS)
def test_ceiling_smoke_and_metadata(benchmark_identifier):
    b = load_benchmark(benchmark_identifier)

    c_direct = b._ceiling_func()
    assert float(c_direct) == c_direct.values  # scalar Score/DataArray
    assert c_direct.attrs.get("ceiling") == "predictor_consistency_split_half"
    assert "error" in c_direct.attrs
    assert float(c_direct.attrs["error"]) >= 0


@pytest.mark.parametrize("benchmark_identifier", BENCHMARKS)
def test_ceiling_cached_matches_direct(benchmark_identifier):
    """
    Requires: you bumped version in benchmark.py after changing ceiling_func.
    Otherwise cached value can reflect an older ceiling.
    """
    b = load_benchmark(benchmark_identifier)
    c_direct = b._ceiling_func()
    c_cached = b.ceiling

    assert float(c_cached) == approx(float(c_direct), abs=3e-3)


@pytest.mark.parametrize("benchmark_identifier", BENCHMARKS)
def test_self_scoring_runs(benchmark_identifier):
    """
    Self-score: feed the benchmark its own assembly as "model features".
    Should be finite and should attach raw/ceiling attrs.
    """
    b = load_benchmark(benchmark_identifier)
    src = b._assembly.copy()
    src = {b._assembly.stimulus_set.identifier: src}
    score = b(PrecomputedFeatures(src, visual_degrees=8))

    assert score is not None
    assert float(score) == float(score.values)
    assert score.attrs.get("ceiling") is not None
    assert score.attrs.get("raw") is not None
