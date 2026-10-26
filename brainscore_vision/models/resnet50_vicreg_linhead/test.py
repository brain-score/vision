"""
Test script for the VicReg model.
"""

# import pytest
import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("resnet50_vicreg_linhead")
    assert model.identifier == "resnet50_vicreg_linhead"
