import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('convnextv2_huge_in22k')
    assert model.identifier == 'convnextv2_huge_in22k'
