import os
import pickle

import pandas as pd
import pytest
from pytest import approx

from brainio_base.assemblies import BehavioralAssembly
from brainscore.assemblies.private import Rajalingham2018Loader
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
        objectome = Rajalingham2018Loader()()
        probabilities = pd.read_pickle(os.path.join(os.path.dirname(__file__),
                                                    f'{model}-probabilities.pkl'))['data']
        probabilities = BehavioralAssembly(probabilities)
        # metric
        i2n = I2n()
        score = i2n(probabilities, objectome)
        score = score.sel(aggregation='center')
        assert score == approx(expected_score, abs=0.005), f"expected {expected_score}, but got {score}"

    def test_ceiling(self):
        objectome = Rajalingham2018Loader()()
        i2n = I2n()
        ceiling = i2n.ceiling(objectome)
        assert ceiling.sel(aggregation='center') == approx(.4786, abs=.0064)
        assert ceiling.sel(aggregation='error') == approx(.00537, abs=.0015)
