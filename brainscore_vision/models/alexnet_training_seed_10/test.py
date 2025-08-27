import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('alexnet_training_seed_10')
    assert model.identifier == 'alexnet_training_seed_10'
    