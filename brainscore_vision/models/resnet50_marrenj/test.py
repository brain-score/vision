"""Minimum sanity test: plugin registers + loads. Brain-Score CI runs this."""
import pytest
import brainscore_vision


@pytest.mark.private_access
def test_has_identifier():
    model = brainscore_vision.load_model('resnet50_marrenj')
    assert model.identifier == 'resnet50_marrenj'
