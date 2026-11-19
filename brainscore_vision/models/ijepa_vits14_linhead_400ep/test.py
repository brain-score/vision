"""
Test script for the I-JEPA ViT-S/14 model.
"""

import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("ijepa_vits14_linhead_400ep")
    assert model.identifier == "ijepa_vits14_linhead_400ep"
