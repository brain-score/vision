from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_finetune_cutmix_e3_robust_linf8255_e0_247x234'] = lambda: ModelCommitment(
    identifier='resnet50_finetune_cutmix_e3_robust_linf8255_e0_247x234',
    activations_model=get_model("resnet50_finetune_cutmix_e3_robust_linf8255_e0_247x234"),
    layers=get_layers("resnet50_finetune_cutmix_e3_robust_linf8255_e0_247x234")
)