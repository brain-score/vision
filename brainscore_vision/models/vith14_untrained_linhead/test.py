"""
Test script for the untrained ViT-H/14 model.
"""

# import pytest
import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("vith14_untrained_linhead")
    assert model.identifier == "vith14_untrained_linhead"
