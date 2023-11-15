import pytest

from brainscore_vision import load_dataset


@pytest.mark.private_access
def test_existence():
    assert load_dataset('ImageNetSlim15000') is not None
