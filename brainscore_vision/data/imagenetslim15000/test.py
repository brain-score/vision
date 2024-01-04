import pytest

from brainscore_vision import load_stimulus_set


@pytest.mark.private_access
def test_existence():
    assert load_stimulus_set('ImageNetSlim15000') is not None
