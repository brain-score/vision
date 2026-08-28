import pytest

import brainscore_vision


@pytest.mark.private_access
def test_existence_stimulus_set():
    assert brainscore_vision.load_stimulus_set('NSDimagesShared2024') is not None

@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('NSD.V1.SharedCombinedSubs.2024', marks=[]),
    pytest.param('NSD.V2.SharedCombinedSubs.2024', marks=[]),
    pytest.param('NSD.V3.SharedCombinedSubs.2024', marks=[]),
    pytest.param('NSD.V4.SharedCombinedSubs.2024', marks=[]),
    pytest.param('NSD.early.SharedCombinedSubs.2024', marks=[]),
    pytest.param('NSD.lateral.SharedCombinedSubs.2024', marks=[]),
    pytest.param('NSD.parietal.SharedCombinedSubs.2024', marks=[]),
    pytest.param('NSD.ventral.SharedCombinedSubs.2024', marks=[]),
])
def test_existence_assembly(assembly_identifier):
    assert brainscore_vision.load_dataset(assembly_identifier) is not None
