import pytest

import brainscore_vision


@pytest.mark.private_access
def test_existence():
    assert brainscore_vision.load_dataset('dicarlo.Kar2019') is not None
