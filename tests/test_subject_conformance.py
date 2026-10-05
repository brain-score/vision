"""Domain adapters implement Subject through the legacy compatibility base."""
from brainscore_core.model_interface import Subject, UnifiedModel
from brainscore_vision.compat.unified_adapter import VisionModelAdapter


def test_vision_adapter_is_subject():
    assert issubclass(VisionModelAdapter, Subject)
    assert issubclass(VisionModelAdapter, UnifiedModel)
