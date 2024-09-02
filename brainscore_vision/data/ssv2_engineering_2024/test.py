import pytest

import brainscore_vision


@pytest.mark.private_access
def test_existence():
    assembly = load_dataset("SSV2ActivityRec2024")
    assert assembly is not None
