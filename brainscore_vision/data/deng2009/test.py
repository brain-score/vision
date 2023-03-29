import pytest
from brainscore_vision import load_stimulus_set


@pytest.mark.private_access
def test_feifei_Deng2009():
    stimulus_set = load_stimulus_set('fei-fei.Deng2009')
    assert len(stimulus_set) == 50_000
    assert len(set(stimulus_set['label'])) == 1_000
