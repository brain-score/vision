"""
Test script for the supervised ResNet-50 model.
"""

# import pytest
import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("resnet50_supervised_linhead")
    assert model.identifier == "resnet50_supervised_linhead"
