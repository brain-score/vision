import pytest

import brainscore_vision


@pytest.mark.private_access
def test_existence():
    assert brainscore_vision.load_stimulus_set('PhysionOCPSmall') is not None

test_existence()
