# Created by David Coggan on 2025 03 13
import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('regnet_y_128gf_E2E')
    assert model.identifier == 'regnet_y_128gf_E2E'