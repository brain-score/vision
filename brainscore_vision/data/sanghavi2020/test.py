import brainscore_vision
import pytest


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('SanghaviMurty2020', marks=[pytest.mark.private_access]),
    pytest.param('SanghaviJozwik2020', marks=[pytest.mark.private_access]),
    pytest.param('Sanghavi2020', marks=[pytest.mark.private_access]),
    pytest.param('SanghaviMurty2020THINGS1', marks=[pytest.mark.private_access]),
    pytest.param('SanghaviMurty2020THINGS2', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert brainscore_vision.load_dataset(assembly_identifier) is not None
