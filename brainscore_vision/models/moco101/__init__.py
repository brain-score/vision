from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry["moco_101_20"] = lambda: ModelCommitment(
    identifier="moco_101_20",
    activations_model=get_model("moco_101_20"),
    layers=LAYERS,
)

model_registry["moco_101_30"] = lambda: ModelCommitment(
    identifier="moco_101_30",
    activations_model=get_model("moco_101_30"),
    layers=LAYERS,
)

model_registry["moco_101_40"] = lambda: ModelCommitment(
    identifier="moco_101_40",
    activations_model=get_model("moco_101_40"),
    layers=LAYERS,
)

model_registry["moco_101_50"] = lambda: ModelCommitment(
    identifier="moco_101_50",
    activations_model=get_model("moco_101_50"),
    layers=LAYERS,
)

model_registry["moco_101_60"] = lambda: ModelCommitment(
    identifier="moco_101_60",
    activations_model=get_model("moco_101_60"),
    layers=LAYERS,
)

model_registry["moco_101_70"] = lambda: ModelCommitment(
    identifier="moco_101_70",
    activations_model=get_model("moco_101_70"),
    layers=LAYERS,
)
