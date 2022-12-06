import brainscore_vision
import pytest

# TODO: add more tests to look at size/contents of assembly


@pytest.mark.parametrize('assembly', (
    'dicarlo.SanghaviMurty2020',
    'dicarlo.SanghaviJozwik2020',
    'dicarlo.Sanghavi2020',
    'dicarlo.SanghaviMurty2020THINGS1',
    'dicarlo.SanghaviMurty2020THINGS2',
))
def test_list_assembly(assembly):
    l = brainscore_vision.list_assemblies()
    assert assembly in l


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('dicarlo.SanghaviMurty2020', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.SanghaviJozwik2020', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Sanghavi2020', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.SanghaviMurty2020THINGS1', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.SanghaviMurty2020THINGS2', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert brainscore_vision.get_assembly(assembly_identifier) is not None
