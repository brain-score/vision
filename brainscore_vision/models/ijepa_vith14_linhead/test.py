"""
Test script for the I-JEPA ViT-H/14 model.
"""

# import pytest
import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("ijepa_vith14_linhead")
    assert model.identifier == "ijepa_vith14_linhead"
