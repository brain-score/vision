import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('custom_model_cv_18_dagger_408')
    assert model.identifier == 'custom_model_cv_18_dagger_408'