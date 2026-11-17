"""
Test script for the ViT-S/14 supervised linear head model.
"""

import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("vits14_supervised_linhead")
    assert model.identifier == "vits14_supervised_linhead"
