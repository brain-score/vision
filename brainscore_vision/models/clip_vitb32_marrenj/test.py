"""Minimum sanity test."""
import pytest
import brainscore_vision


@pytest.mark.private_access
def test_has_identifier():
    model = brainscore_vision.load_model('clip_vitb32_marrenj')
    assert model.identifier == 'clip_vitb32_marrenj'
