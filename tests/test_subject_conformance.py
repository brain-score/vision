"""v1.5: the vision adapter conforms to the Subject interface.

``UnifiedModel`` was renamed to ``Subject`` in v1.5 (with ``UnifiedModel`` kept
as a deprecated alias). The vision adapter now subclasses ``Subject``; this test
confirms the conformance and that the alias identity still holds.
"""
from brainscore_core.model_interface import Subject, UnifiedModel
from brainscore_vision.compat.unified_adapter import VisionModelAdapter


def test_vision_adapter_is_subject():
    assert issubclass(VisionModelAdapter, Subject)
    assert issubclass(VisionModelAdapter, UnifiedModel)  # alias identity
