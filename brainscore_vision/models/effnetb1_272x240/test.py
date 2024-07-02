# Left empty as part of 2023 models migration
import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('effnetb1_272x240')
    assert model.identifier == 'effnetb1_272x240'