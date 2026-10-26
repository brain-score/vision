"""
Test script for the Resnet-50 barlow-twins model.
"""

# import pytest
import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("resnet50_barlowtwins_linhead")
    assert model.identifier == "resnet50_barlowtwins_linhead"
