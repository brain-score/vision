from pathlib import Path

import numpy as np
import pytest
from numpy.random.mtrand import RandomState
from pytest import approx

from brainio.assemblies import NeuroidAssembly, DataAssembly
from brainscore_vision import load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.benchmark_helpers.test_helper import VisualDegreesTests, NumberOfTrialsTests
from brainscore_vision.benchmarks.kar2019 import DicarloKar2019OST
from brainscore_vision.data_helpers import s3

visual_degrees = VisualDegreesTests()
number_trials = NumberOfTrialsTests()


@pytest.mark.memory_intense
@pytest.mark.private_access
@pytest.mark.slow
def test_Kar2019ost_cornet_s():
    benchmark = load_benchmark('dicarlo.Kar2019-ost')
    filename = 'cornet_s-kar2019.nc'
    filepath = Path(__file__).parent / filename
    s3.download_file_if_not_exists(local_path=filepath,
                                   bucket='brain-score-tests', remote_filepath=f'tests/test_benchmarks/{filename}')
    precomputed_features = NeuroidAssembly.from_files(
        filepath,
        stimulus_set_identifier=benchmark._assembly.stimulus_set.identifier,
        stimulus_set=benchmark._assembly.stimulus_set)
    precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
    # score
    score = benchmark(precomputed_features).raw
    assert score == approx(.316, abs=.005)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark, candidate_degrees, image_id, expected', [
    pytest.param('dicarlo.Kar2019-ost', 14, '6d19b24c29832dfb28360e7731e3261c13a4287f',
                 approx(.225021, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Kar2019-ost', 6, '6d19b24c29832dfb28360e7731e3261c13a4287f',
                 approx(.001248, abs=.0001), marks=[pytest.mark.private_access]),
])
def test_amount_gray(benchmark: str, candidate_degrees: int, image_id: str, expected: float):
    visual_degrees.amount_gray_test(benchmark, candidate_degrees, image_id, expected)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', ['dicarlo.Kar2019-ost'])
def test_repetitions(benchmark_identifier):
    number_trials.repetitions_test(benchmark_identifier)


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_no_time():
    benchmark = DicarloKar2019OST()
    rnd = RandomState(0)
    stimuli = benchmark._assembly.stimulus_set
    source = DataAssembly(rnd.rand(len(stimuli), 5, 1), coords={
        'stimulus_id': ('presentation', stimuli['stimulus_id']),
        'image_label': ('presentation', stimuli['image_label']),
        'truth': ('presentation', stimuli['truth']),
        'neuroid_id': ('neuroid', list(range(5))),
        'layer': ('neuroid', ['test'] * 5),
        'time_bin_start': ('time_bin', [70]),
        'time_bin_end': ('time_bin', [170]),
    }, dims=['presentation', 'neuroid', 'time_bin'])
    source.name = __name__ + ".test_notime"
    score = benchmark(PrecomputedFeatures(source, visual_degrees=8))
    assert np.isnan(score)  # not a temporal model
    assert np.isnan(score.raw)  # not a temporal model
    assert score.attrs['ceiling'] == approx(.79)


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_random_time():
    benchmark = DicarloKar2019OST()
    rnd = RandomState(0)
    stimuli = benchmark._assembly.stimulus_set
    source = DataAssembly(rnd.rand(len(stimuli), 5, 5), coords={
        'stimulus_id': ('presentation', stimuli['stimulus_id']),
        'image_label': ('presentation', stimuli['image_label']),
        'truth': ('presentation', stimuli['truth']),
        'neuroid_id': ('neuroid', list(range(5))),
        'layer': ('neuroid', ['test'] * 5),
        'time_bin_start': ('time_bin', [70, 90, 110, 130, 150]),
        'time_bin_end': ('time_bin', [90, 110, 130, 150, 170]),
    }, dims=['presentation', 'neuroid', 'time_bin'])
    source.name = __name__ + ".test_notime"
    score = benchmark(PrecomputedFeatures(source, visual_degrees=8))
    assert np.isnan(score)  # not a good temporal model
