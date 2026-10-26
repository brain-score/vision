"""
Test script for the Resnet-50 simsiam model.
"""

# import pytest
import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("resnet50_simsiam_linhead")
    assert model.identifier == "resnet50_simsiam_linhead"
