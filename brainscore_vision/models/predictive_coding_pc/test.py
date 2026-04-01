import pytest
import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model('predictive_coding_pc')
    assert model.identifier == 'predictive_coding_pc'
