import brainscore_vision
import pytest

# TODO: add more tests to look at size/contents of assembly


@pytest.mark.parametrize('assembly', (
    'dicarlo.Rajalingham2018.public',
    'dicarlo.Rajalingham2018.private',
))
def test_list_assembly(assembly):
    l = brainscore_vision.list_assemblies()
    assert assembly in l


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('dicarlo.Rajalingham2018.public', marks=[]),
    pytest.param('dicarlo.Rajalingham2018.private', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert brainscore_vision.get_assembly(assembly_identifier) is not None
