"""
Test script for the ResNet-50 untrained with a trained linear head model.
"""

# import pytest
import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("resnet50_untrained_linhead")
    assert model.identifier == "resnet50_untrained_linhead"
