from pathlib import Path

import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import load_metric
from brainscore_vision.benchmarks.rajalingham2018.benchmarks.benchmark import load_assembly


@pytest.mark.private_access
class TestI2N:
    @pytest.mark.parametrize(['model', 'expected_score'], [
        ('alexnet', .253),
        ('resnet34', .37787),
        ('resnet18', .3638),
    ])
    def test_model(self, model, expected_score):
        # assemblies
        objectome = load_assembly()
        probabilities = Path(__file__).parent / 'test_resources' / f'{model}-probabilities.nc'
        probabilities = BehavioralAssembly.from_files(
            probabilities,
            stimulus_set_identifier=objectome.attrs['stimulus_set_identifier'],
            stimulus_set=objectome.attrs['stimulus_set'])
        # metric
        i2n = load_metric('i2n')
        score = i2n(probabilities, objectome)
        assert score == approx(expected_score, abs=0.005), f"expected {expected_score}, but got {score}"

    def test_ceiling(self):
        objectome = load_assembly()
        i2n = load_metric('i2n')
        ceiling = i2n.ceiling(objectome)
        assert ceiling == approx(.4786, abs=.0064)
        assert ceiling.error == approx(.00537, abs=.0015)
