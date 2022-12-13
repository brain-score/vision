import numpy as np
import os
from pathlib import Path

import pandas as pd
import xarray as xr
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision.benchmarks.rajalingham2018 import DicarloRajalingham2018I2n
from brainscore_vision.model_interface import BrainModel
from brainscore_vision import benchmark_registry
from todotests.test_benchmarks import PrecomputedFeatures
from brainscore_vision.benchmarks.test_helper import TestVisualDegrees, TestNumberOfTrials, TestBenchmarkRegistry

visual_degrees = TestVisualDegrees()
number_trials = TestNumberOfTrials()
in_registry = TestBenchmarkRegistry()


@pytest.mark.parametrize('benchmark', [
    'dicarlo.Rajalingham2018-i2n',
    'dicarlo.Rajalingham2018public-i2n',
])
def test_benchmark_registry(benchmark):
    in_registry.benchmark_in_registry(benchmark)


@pytest.mark.private_access
class TestRajalingham2018:
    def test_ceiling(self):
        benchmark = DicarloRajalingham2018I2n()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.479, abs=.0064)

    @pytest.mark.parametrize(['model', 'expected_score'],
                             [
                                 ('alexnet', .253),
                                 ('resnet34', .37787),
                                 ('resnet18', .3638),
                             ])
    def test_precomputed(self, model, expected_score):
        benchmark = DicarloRajalingham2018I2n()
        probabilities = Path(__file__).parent.parent / 'test_metrics' / f'{model}-probabilities.nc'
        probabilities = BehavioralAssembly.from_files(probabilities, stimulus_set_identifier=benchmark._assembly.stimulus_set.identifier, stimulus_set=benchmark._assembly.stimulus_set)
        candidate = PrecomputedProbabilities(probabilities)
        score = benchmark(candidate)
        assert score.raw.sel(aggregation='center') == approx(expected_score, abs=.005)
        assert score.sel(aggregation='center') == approx(expected_score / np.sqrt(.479), abs=.005)


class PrecomputedProbabilities(BrainModel):
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def visual_degrees(self) -> int:
        return 8

    def start_task(self, task: BrainModel.Task, fitting_stimuli):
        assert task == BrainModel.Task.probabilities
        assert len(fitting_stimuli) == 2160

    def look_at(self, stimuli, number_of_trials=1):
        assert set(self.probabilities['stimulus_id'].values) == set(stimuli['stimulus_id'].values)
        image_ids = self.probabilities['stimulus_id'].values.tolist()
        probabilities = self.probabilities[[image_ids.index(image_id) for image_id in stimuli['stimulus_id'].values], :]
        assert all(probabilities['stimulus_id'].values == stimuli['stimulus_id'].values)
        return probabilities


# TODO: question arises again w/ public
def test_Rajalingham2018public():
    benchmark = benchmark_registry['dicarlo.Rajalingham2018public-i2n']
    # load features
    precomputed_features = Path(__file__).parent / 'CORnetZ-rajalingham2018public.nc'
    precomputed_features = BehavioralAssembly.from_files(
        precomputed_features,
        stimulus_set_identifier=benchmark._assembly.stimulus_set.identifier,
        stimulus_set=benchmark._assembly.stimulus_set)
    precomputed_features = PrecomputedFeatures(precomputed_features,
                                               visual_degrees=8,  # doesn't matter, features are already computed
                                               )
    # score
    score = benchmark(precomputed_features).raw
    assert score.sel(aggregation='center') == approx(.136923, abs=.005)


@pytest.mark.parametrize('benchmark, candidate_degrees, image_id, expected', [
    pytest.param('dicarlo.Rajalingham2018-i2n', 14, '0223bf9e5db0edad21976b16494fe9396a5ef145',
                 approx(.225023, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Rajalingham2018-i2n', 6, '0223bf9e5db0edad21976b16494fe9396a5ef145',
                 approx(.002244, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Rajalingham2018public-i2n', 14, '0020cef91bd626e9fbbabd853494ee444e5c9ecb',
                 approx(.22486, abs=.0001), marks=[]),
    pytest.param('dicarlo.Rajalingham2018public-i2n', 6, '0020cef91bd626e9fbbabd853494ee444e5c9ecb',
                 approx(.00097, abs=.0001), marks=[]),
])
def test_amount_gray(benchmark, candidate_degrees, image_id, expected, brainio_home, resultcaching_home,
                     brainscore_home):
    visual_degrees.amount_gray_test(benchmark, candidate_degrees, image_id, expected, brainio_home,
                                    resultcaching_home, brainscore_home)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', ['dicarlo.Rajalingham2018-i2n'])
def test_repetitions(benchmark_identifier):
    number_trials.repetitions_test(benchmark_identifier)
