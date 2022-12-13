import brainscore_vision
import pytest

# TODO: add more tests to look at size/contents of assembly


@pytest.mark.parametrize('assembly', (
    'aru.Kuzovkin2018',
    'aru.Kuzovkin2018',
))
def test_list_assembly(assembly):
    l = brainscore_vision.list_assemblies()
    assert assembly in l


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('aru.Kuzovkin2018', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert brainscore_vision.get_assembly(assembly_identifier) is not None