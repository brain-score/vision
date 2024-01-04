import pytest
from brainscore_vision import load_stimulus_set


@pytest.mark.private_access
def test_num_elements():
    stimulus_set = load_stimulus_set('imagenet_val')
    assert len(stimulus_set) == 50_000
    assert len(set(stimulus_set['label'])) == 1_000
