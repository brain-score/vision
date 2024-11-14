import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('sam_test_resnet_4')
    assert model.identifier == 'sam_test_resnet_4'
