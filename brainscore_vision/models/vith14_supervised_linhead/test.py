"""
Test script for the ViT-H/14 supervised linear head model.
"""

import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("vith14_supervised_linhead")
    assert model.identifier == "vith14_supervised_linhead"
