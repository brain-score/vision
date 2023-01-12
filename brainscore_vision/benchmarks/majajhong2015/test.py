import pytest
from pytest import approx

from brainscore_vision.benchmark_helpers.test_helper import TestStandardized, TestPrecomputed, TestNumberOfTrials, \
    TestBenchmarkRegistry, TestVisualDegrees
from brainscore_vision.benchmarks.majajhong2015.benchmark import MajajHongV4PublicBenchmark, MajajHongITPublicBenchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures

# should these be in function definitions
standardized_tests = TestStandardized()
precomputed_test = TestPrecomputed()
num_trials_test = TestNumberOfTrials()
registry_test = TestBenchmarkRegistry()
visual_degrees = TestVisualDegrees()


@pytest.mark.parametrize('benchmark', [

])
def test_benchmark_registry(benchmark):
    registry_test.benchmark_in_registry(benchmark)


@pytest.mark.parametrize('benchmark, expected', [
    pytest.param('dicarlo.MajajHong2015.V4-pls', approx(.89503, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.MajajHong2015.IT-pls', approx(.821841, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.MajajHong2015.V4-rdm', approx(.936473, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.MajajHong2015.IT-rdm', approx(.887618, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_ceilings(benchmark, expected):
    standardized_tests.ceilings_test(benchmark, expected)


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('dicarlo.MajajHong2015.V4-pls', 8, approx(.923713, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.MajajHong2015.IT-pls', 8, approx(.823433, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.self_regression_test(benchmark, visual_degrees, expected)


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('dicarlo.MajajHong2015.V4-rdm', 8, approx(1, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.MajajHong2015.IT-rdm', 8, approx(1, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_self_rdm(benchmark: str, visual_degrees: int, expected: float):
    standardized_tests.self_rdm_test(benchmark, visual_degrees, expected)


@pytest.mark.memory_intense
@pytest.mark.parametrize('benchmark, expected', [
    ('dicarlo.MajajHong2015.V4-pls', approx(.490236, abs=.005)),
    ('dicarlo.MajajHong2015.IT-pls', approx(.584053, abs=.005)),
])
def test_MajajHong2015(benchmark, expected):
    precomputed_test.run_test(benchmark=benchmark, file='alexnet-majaj2015.private-features.12.nc', expected=expected)


@pytest.mark.parametrize('benchmark, candidate_degrees, image_id, expected', [
    pytest.param('dicarlo.MajajHong2015.V4-pls', 14, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                 approx(.251345, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.MajajHong2015.V4-pls', 6, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                 approx(.0054886, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.MajajHong2015public.V4-pls', 14, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                 approx(.25071, abs=.0001), marks=[]),
    pytest.param('dicarlo.MajajHong2015public.V4-pls', 6, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                 approx(.00460, abs=.0001), marks=[]),
    pytest.param('dicarlo.MajajHong2015.IT-pls', 14, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                 approx(.251345, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.MajajHong2015.IT-pls', 6, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                 approx(.0054886, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.MajajHong2015public.IT-pls', 14, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                 approx(.25071, abs=.0001), marks=[]),
    pytest.param('dicarlo.MajajHong2015public.IT-pls', 6, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                 approx(.00460, abs=.0001), marks=[]),
])
def test_amount_gray(benchmark, candidate_degrees, image_id, expected, brainio_home, resultcaching_home,
                     brainscore_home):
    visual_degrees.amount_gray_test(benchmark, candidate_degrees, image_id, expected, brainio_home,
                                    resultcaching_home, brainscore_home)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', [
    # V4
    'dicarlo.MajajHong2015.V4-pls',
    # IT
    'dicarlo.MajajHong2015.IT-pls',
])
def test_repetitions(benchmark_identifier):
    num_trials_test.repetitions_test(benchmark_identifier)

@pytest.mark.parametrize('benchmark_ctr, visual_degrees, expected', [
    pytest.param(MajajHongV4PublicBenchmark, 8, approx(.897956, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param(MajajHongITPublicBenchmark, 8, approx(.816251, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_self(benchmark_ctr, visual_degrees, expected):
    benchmark = benchmark_ctr()
    source = benchmark._assembly.copy()
    source = {benchmark._assembly.stimulus_set.identifier: source}
    score = benchmark(PrecomputedFeatures(source, visual_degrees=visual_degrees)).raw
    assert score.sel(aggregation='center') == expected
