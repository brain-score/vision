import pytest
from pathlib import Path
from pytest import approx

from brainscore_vision import benchmark_registry
from brainio.assemblies import NeuroidAssembly
from tests.test_benchmarks import PrecomputedFeatures
from brainscore_vision.benchmarks.test_helper import TestVisualDegrees, TestNumberOfTrials

visual_degrees = TestVisualDegrees()
number_trials = TestNumberOfTrials()

@pytest.mark.memory_intense
@pytest.mark.private_access
@pytest.mark.slow
def test_Kar2019ost_cornet_s():
    benchmark = benchmark_registry['dicarlo.Kar2019-ost']
    # might have to change parent here
    precomputed_features = Path(__file__).parent / 'cornet_s-kar2019.nc'
    precomputed_features = NeuroidAssembly.from_files(
        precomputed_features,
        stimulus_set_identifier=benchmark._assembly.stimulus_set.identifier,
        stimulus_set=benchmark._assembly.stimulus_set)
    precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
    # score
    score = benchmark(precomputed_features).raw
    assert score.sel(aggregation='center') == approx(.316, abs=.005)


@pytest.mark.parametrize('benchmark, candidate_degrees, image_id, expected', [
    pytest.param('dicarlo.Kar2019-ost', 14, '6d19b24c29832dfb28360e7731e3261c13a4287f',
                 approx(.225021, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Kar2019-ost', 6, '6d19b24c29832dfb28360e7731e3261c13a4287f',
                 approx(.001248, abs=.0001), marks=[pytest.mark.private_access]),
])
def test_amount_gray(benchmark, candidate_degrees, image_id, expected, brainio_home, resultcaching_home,
                     brainscore_home):
    visual_degrees.test_amount_gray(benchmark, candidate_degrees, image_id, expected, brainio_home,
                                    resultcaching_home, brainscore_home)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', ['dicarlo.Kar2019-ost'])
def test_repetitions(benchmark_identifier):
    number_trials.test_repetitions(benchmark_identifier)
