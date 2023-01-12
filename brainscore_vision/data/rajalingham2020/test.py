import brainscore_vision
import pytest


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('dicarlo.Rajalingham2020', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert brainscore_vision.load_dataset(assembly_identifier) is not None
