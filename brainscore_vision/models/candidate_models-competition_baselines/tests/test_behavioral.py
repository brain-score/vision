import pytest
from pytest import approx

from brainscore import score_model
from candidate_models.model_commitments import brain_translated_pool


@pytest.mark.memory_intense
@pytest.mark.private_access
@pytest.mark.parametrize(['model_identifier', 'expected_score'],
                         [
                             ('alexnet', .253),
                             ('resnet-34', .37787),
                             ('resnet-18', .3638),
                         ])
def test_model(model_identifier, expected_score):
    model = brain_translated_pool[model_identifier]
    score = score_model(model_identifier=model_identifier, model=model,
                        benchmark_identifier='dicarlo.Rajalingham2018-i2n')
    assert score.raw.sel(aggregation='center') == approx(expected_score, abs=.005)
