import pytest
import brainscore_vision

def test_has_identifier():
    model = brainscore_vision.load_model('chresmax_v3_cl_01_v2')
    assert model.identifier == 'chresmax_v3_cl_01_v2'
