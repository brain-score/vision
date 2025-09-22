# Left empty as part of 2023 models migration

import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('AdvProp_efficientnet-b6')
    assert model.identifier == 'AdvProp_efficientnet-b6'