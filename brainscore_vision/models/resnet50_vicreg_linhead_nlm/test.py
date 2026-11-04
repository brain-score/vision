"""
Test script for the VicReg model without a fixed layer mapping.
"""

import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("resnet50_vicreg_linhead_nlm")
    assert model.identifier == "resnet50_vicreg_linhead_nlm"
