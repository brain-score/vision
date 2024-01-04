import pytest

from brainscore_vision import load_dataset


@pytest.mark.private_access
def test_existence():
    assert load_dataset('Cichy2019') is not None
