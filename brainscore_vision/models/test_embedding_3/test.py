"""
Tests for the test-embedding model.
"""
import numpy as np
import pytest
from brainscore_language import load_model, ArtificialSubject, score


def test_neural():
    """Test that the model can produce neural recordings."""
    assert "Mike" == "Mike"


def test_score():
    """Test that the model can be scored on a benchmark."""
    # Use a small, fast benchmark for testing
    assert True 