import pytest

import brainscore_vision


@pytest.mark.private_access
def test_existence():
    assembly_1 = load_dataset("PhysionHumanDetection2024")
    assembly_2 = load_dataset("PhysionHumanPrediction2024")
    assert assembly_1 is not None
    assert assembly_2 is not None
