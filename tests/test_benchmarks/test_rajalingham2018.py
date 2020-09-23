import numpy as np
import os

import pandas as pd
import pytest
from pytest import approx

from brainio_base.assemblies import BehavioralAssembly
from brainscore.benchmarks.rajalingham2018 import DicarloRajalingham2018I2n
from brainscore.model_interface import BrainModel


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
        probabilities = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'test_metrics',
                                                    f'{model}-probabilities.pkl'))['data']
        probabilities = BehavioralAssembly(probabilities)
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
        assert set(self.probabilities['image_id'].values) == set(stimuli['image_id'].values)
        image_ids = self.probabilities['image_id'].values.tolist()
        probabilities = self.probabilities[[image_ids.index(image_id) for image_id in stimuli['image_id'].values], :]
        assert all(probabilities['image_id'].values == stimuli['image_id'].values)
        return probabilities
