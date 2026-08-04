from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

try:
    from .model import get_bibtex, get_layers, get_model, get_model_list
except ImportError:
    import importlib.util
    from pathlib import Path

    _MODEL_PATH = Path(__file__).resolve().parent / "model.py"
    _MODEL_SPEC = importlib.util.spec_from_file_location(f"{__name__}_model", _MODEL_PATH)
    if _MODEL_SPEC is None or _MODEL_SPEC.loader is None:
        raise ImportError(f"Unable to load model module from {_MODEL_PATH}")
    _MODEL_MODULE = importlib.util.module_from_spec(_MODEL_SPEC)
    _MODEL_SPEC.loader.exec_module(_MODEL_MODULE)
    get_bibtex = _MODEL_MODULE.get_bibtex
    get_layers = _MODEL_MODULE.get_layers
    get_model = _MODEL_MODULE.get_model
    get_model_list = _MODEL_MODULE.get_model_list

__all__ = ["get_bibtex", "get_layers", "get_model", "get_model_list"]

model_registry['20260408_2033_stl10_disentangled_alpha04'] = lambda: ModelCommitment(
    identifier='20260408_2033_stl10_disentangled_alpha04',
    activations_model=get_model('20260408_2033_stl10_disentangled_alpha04'),
    layers=get_layers('20260408_2033_stl10_disentangled_alpha04'),
    visual_degrees=8,
)
