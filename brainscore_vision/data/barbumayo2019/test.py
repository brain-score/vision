import pytest
from brainscore_vision import load_stimulus_set


@pytest.mark.private_access
def test_Katz_BarbuMayo2019():
    stimulus_set = load_stimulus_set('katz.BarbuMayo2019')
    assert len(stimulus_set) == 17261
    assert len(set(stimulus_set['synset'])) == 104
