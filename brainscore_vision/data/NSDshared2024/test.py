import pytest

import brainscore_vision


@pytest.mark.private_access
def test_existence_stimulus_set():
    assert brainscore_vision.load_stimulus_set('NSDimagesShared2024') is not None

@pytest.mark.private_access
def test_existence_assembly():
    assembly = brainscore_vision.load_dataset("NSD.V1.SharedCombinedSubs.2024")
    assert assembly is not None

test_existence_assembly()