import pytest
from brainscore_vision import load_dataset


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('aru.Cichy2019', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None
