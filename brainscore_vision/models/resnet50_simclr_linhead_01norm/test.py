"""
Test script for the Resnet-50 SimCLR model with a linear head and 0-1 normalization.
"""

import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("resnet50_simclr_linhead_01norm")
    assert model.identifier == "resnet50_simclr_linhead_01norm"
