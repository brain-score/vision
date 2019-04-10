import os
import pickle

import pandas as pd
import pytest
from pytest import approx

from brainio_base.assemblies import BehavioralAssembly
from brainscore.metrics.behavior import I2n


class TestI2N:
    @pytest.mark.parametrize(['model', 'expected_score'],
                             [
                                 ('alexnet', .253),
                                 ('resnet34', .37787),
                                 ('resnet18', .3638),
                             ])
    def test_model(self, model, expected_score):
        # assemblies
        testing_objectome = self.get_objectome('full_trials')
        feature_responses = pd.read_pickle(os.path.join(os.path.dirname(__file__),
                                                        f'{model}-probabilities.pkl'))['data']
        feature_responses = BehavioralAssembly(feature_responses)
        # metric
        i2n = I2n()
        score = i2n(feature_responses, testing_objectome)
        score = score.sel(aggregation='center')
        assert score == approx(expected_score, abs=0.005), f"expected {expected_score}, but got {score}"

    def test_ceiling(self):
        objectome = self.get_objectome('full_trials')
        i2n = I2n()
        ceiling = i2n.ceiling(objectome)
        assert ceiling.sel(aggregation='center') == approx(.4786, abs=.0064)
        assert ceiling.sel(aggregation='error') == approx(.00537, abs=.0015)

    def get_objectome(self, subtype):
        with open(f'/braintree/home/msch/brainio_contrib/mkgu_packaging/dicarlo/dicarlo.Rajalingham2018.{subtype}.pkl',
                  'rb') as f:
            objectome = pickle.load(f)
        return objectome
