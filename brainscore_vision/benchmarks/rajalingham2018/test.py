import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import benchmark_registry, load_benchmark, load_metric
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.benchmark_helpers.test_helper import TestVisualDegrees, TestNumberOfTrials
from brainscore_vision.benchmarks.rajalingham2018 import DicarloRajalingham2018I2n
from brainscore_vision.benchmarks.rajalingham2018.benchmarks.benchmark import _DicarloRajalingham2018
from brainscore_vision.data_helpers import s3
from brainscore_vision.model_helpers.brain_transformation import ProbabilitiesMapping
from brainscore_vision.model_interface import BrainModel

visual_degrees = TestVisualDegrees()
number_trials = TestNumberOfTrials()


@pytest.mark.parametrize('benchmark', [
    'dicarlo.Rajalingham2018-i2n',
    'dicarlo.Rajalingham2018public-i2n',
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry


@pytest.mark.private_access
class TestRajalingham2018:
    def test_ceiling(self):
        benchmark = DicarloRajalingham2018I2n()
        ceiling = benchmark.ceiling
        assert ceiling == approx(.479, abs=.0064)

    @pytest.mark.parametrize(['model', 'expected_score'],
                             [
                                 ('alexnet', .253),
                                 ('resnet34', .37787),
                                 ('resnet18', .3638),
                             ])
    def test_precomputed(self, model, expected_score):
        benchmark = DicarloRajalingham2018I2n()
        probabilities = Path(__file__).parent.parent / 'test_metrics' / f'{model}-probabilities.nc'
        probabilities = BehavioralAssembly.from_files(probabilities,
                                                      stimulus_set_identifier=benchmark._assembly.stimulus_set.identifier,
                                                      stimulus_set=benchmark._assembly.stimulus_set)
        candidate = PrecomputedProbabilities(probabilities)
        score = benchmark(candidate)
        assert score.raw == approx(expected_score, abs=.005)
        assert score == approx(expected_score / np.sqrt(.479), abs=.005)


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


def test_Rajalingham2018public():
    benchmark = load_benchmark('dicarlo.Rajalingham2018public-i2n')
    # load features
    filename = 'CORnetZ-rajalingham2018public.nc'
    filepath = Path(__file__).parent / filename
    s3.download_file_if_not_exists(filepath,
                                   bucket='brain-score-tests', remote_filepath=f'tests/test_benchmarks/{filename}')
    precomputed_features = BehavioralAssembly.from_files(
        filepath,
        stimulus_set_identifier=benchmark._assembly.stimulus_set.identifier,
        stimulus_set=benchmark._assembly.stimulus_set)
    precomputed_features = PrecomputedFeatures(precomputed_features,
                                               visual_degrees=8,  # doesn't matter, features are already computed
                                               )
    # score
    score = benchmark(precomputed_features).raw
    assert score == approx(.136923, abs=.005)


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


@pytest.mark.private_access
class TestMetricScore:
    @pytest.mark.parametrize(['model', 'expected_score'],
                             [
                                 ('alexnet', .253),
                                 ('resnet34', .37787),
                                 ('resnet18', .3638),
                             ])
    def test_model(self, model, expected_score):
        class UnceiledBenchmark(_DicarloRajalingham2018):
            def __init__(self):
                metric = load_metric('i2n')
                super(UnceiledBenchmark, self).__init__(metric=metric, metric_identifier='i2n')

            def __call__(self, candidate: BrainModel):
                candidate.start_task(BrainModel.Task.probabilities, self._fitting_stimuli)
                probabilities = candidate.look_at(self._assembly.stimulus_set)
                score = self._metric(probabilities, self._assembly)
                return score

        benchmark = UnceiledBenchmark()
        # features
        feature_responses = xr.load_dataarray(Path(__file__).parent / 'test_resources' /
                                              f'identifier={model},stimuli_identifier=objectome-240.nc')
        feature_responses['stimulus_id'] = 'stimulus_path', [os.path.splitext(os.path.basename(path))[0]
                                                             for path in feature_responses['stimulus_path'].values]
        feature_responses = feature_responses.stack(presentation=['stimulus_path'])
        assert len(np.unique(feature_responses['layer'])) == 1  # only penultimate layer

        class PrecomputedFeatures:
            def __init__(self, precomputed_features):
                self.features = precomputed_features

            def __call__(self, stimuli, layers):
                np.testing.assert_array_equal(layers, ['behavioral-layer'])
                self_stimulus_ids = self.features['stimulus_id'].values.tolist()
                indices = [self_stimulus_ids.index(stimulus_id) for stimulus_id in stimuli['stimulus_id'].values]
                features = self.features[{'presentation': indices}]
                return features

        # evaluate candidate
        transformation = ProbabilitiesMapping(identifier=f'TestI2N.{model}',
                                              activations_model=PrecomputedFeatures(feature_responses),
                                              layer='behavioral-layer')
        score = benchmark(transformation)
        score = score.sel(aggregation='center')
        assert score == approx(expected_score, abs=0.005), f"expected {expected_score}, but got {score}"
