import pytest
import brainscore_vision

def test_has_identifier():
    model = brainscore_vision.load_model('hmax_v3_adj')
    assert model.identifier == 'hmax_v3_adj'
