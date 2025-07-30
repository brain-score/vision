import pytest
import brainscore_vision

def test_has_identifier():
    model = brainscore_vision.get_model('chresmax_v3_cl_01')
    assert model.identifier == 'chresmax_v3_cl_01'
