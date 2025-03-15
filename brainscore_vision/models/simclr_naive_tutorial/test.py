import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('simclr_naive_tutorial')
    assert model.identifier == 'simclr_naive_tutorial'
