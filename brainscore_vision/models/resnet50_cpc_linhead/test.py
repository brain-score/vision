"""
Test script for the Resnet-50 CPC model.
"""

import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("resnet50_cpc_linhead")
    assert model.identifier == "resnet50_cpc_linhead"
