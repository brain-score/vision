import pytest

import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('seresnext50_32x4d')
    assert model.identifier == 'seresnext50_32x4d'
