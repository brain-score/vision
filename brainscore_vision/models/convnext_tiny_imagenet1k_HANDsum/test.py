import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('convnext_tiny_imagenet1k_HANDsum')
    assert model.identifier == 'convnext_tiny_imagenet1k_HANDsum'
