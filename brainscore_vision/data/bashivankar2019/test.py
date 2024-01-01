import numpy as np
import pytest

import brainscore_vision


@pytest.mark.private_access
@pytest.mark.parametrize('assembly_identifier', [
    'BashivanKar2019.naturalistic',
    'BashivanKar2019.synthetic',
])
def test_existence(assembly_identifier):
    assert brainscore_vision.load_dataset(assembly_identifier) is not None


@pytest.mark.private_access
@pytest.mark.parametrize('assembly,shape,nans', [
    ('BashivanKar2019.naturalistic', (24320, 233, 1), 309760),
    ('BashivanKar2019.synthetic', (21360, 233, 1), 4319940),
])
def test_assembly_shape(assembly, shape, nans):
    assy = brainscore_vision.load_dataset(assembly)
    assert assy.shape == shape
    assert np.count_nonzero(np.isnan(assy)) == nans
