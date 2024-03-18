from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry["ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V1"] = lambda: ModelCommitment(
    identifier="ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V1",
    activations_model=get_model("ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V1"),
    layers=LAYERS["ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V1"],
)

model_registry["ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V2"] = lambda: ModelCommitment(
    identifier="ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V2",
    activations_model=get_model("ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V2"),
    layers=LAYERS["ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V2"],
)

model_registry["ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V4"] = lambda: ModelCommitment(
    identifier="ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V4",
    activations_model=get_model("ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V4"),
    layers=LAYERS["ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V4"],
)

model_registry["ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-IT"] = lambda: ModelCommitment(
    identifier="ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-IT",
    activations_model=get_model("ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-IT"),
    layers=LAYERS["ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-IT"],
)
