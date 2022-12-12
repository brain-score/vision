import brainscore_vision
import brainio
import pytest
import numpy as np

# TODO: add more tests to look at size/contents of assembly


@pytest.mark.parametrize('assembly', (
    'dicarlo.BashivanKar2019.naturalistic',
    'dicarlo.BashivanKar2019.synthetic',
))
def test_list_assembly(assembly):
    l = brainscore_vision.list_assemblies()
    assert assembly in l


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('dicarlo.BashivanKar2019.naturalistic', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.BashivanKar2019.synthetic', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert brainscore_vision.get_assembly(assembly_identifier) is not None


@pytest.mark.parametrize('assembly,shape,nans', [
    pytest.param('dicarlo.BashivanKar2019.naturalistic', (24320, 233, 1), 309760, marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.BashivanKar2019.synthetic', (21360, 233, 1), 4319940, marks=[pytest.mark.private_access]),
])
def test_assembly_shape(assembly, shape, nans):
    assy = brainio.get_assembly(assembly)
    assert assy.shape == shape
    assert np.count_nonzero(np.isnan(assy)) == nans
