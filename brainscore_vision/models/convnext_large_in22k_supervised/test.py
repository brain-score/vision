import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('convnext_large_in22k_supervised')
    assert model.identifier == 'convnext_large_in22k_supervised'
