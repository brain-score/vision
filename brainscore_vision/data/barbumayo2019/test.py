import pytest

from brainscore_vision import load_stimulus_set


@pytest.mark.private_access
def test_num_elements():
    stimulus_set = load_stimulus_set('BarbuMayo2019')
    assert len(stimulus_set) == 17261
    assert len(set(stimulus_set['synset'])) == 104
