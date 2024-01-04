import pytest

from brainscore_vision import load_dataset


@pytest.mark.private_access
def test_existence():
    assert load_dataset('David2004') is not None
