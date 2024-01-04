import pytest

import brainscore_vision


@pytest.mark.private_access
def test_existence():
    assert brainscore_vision.load_dataset('Rajalingham2020') is not None
