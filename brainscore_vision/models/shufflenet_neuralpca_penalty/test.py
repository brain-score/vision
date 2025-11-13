import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('shufflenet_neuralpca_penalty')
    assert model.identifier == 'shufflenet_neuralpca_penalty'
