"""
Test script for the I-JEPA ViT-H/14 model without a fixed layer mapping.
"""

import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("ijepa_vith14_linhead_nlm")
    assert model.identifier == "ijepa_vith14_linhead_nlm"
