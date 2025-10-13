import functools
import os
import warnings
import timm
from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper, load_preprocess_images
)
from torch.hub import load_state_dict_from_url

# Optional: if you have a custom ViT-S weight URL, set it here
VITPOSE_S_URL = os.environ.get("VITPOSE_S_URL", "")  # empty means: skip

def _maybe_load_custom_weights(vit):
    if not VITPOSE_S_URL:
        return
    try:
        sd = load_state_dict_from_url(VITPOSE_S_URL, map_location="cpu", check_hash=False)
        sd = sd.get("state_dict", sd)
        vit.load_state_dict(sd, strict=False)
    except Exception as e:
        warnings.warn(f"[vitpose_s_backbone] Skipping weight download in CI: {e}")

def get_model():
    # IMPORTANT: avoid timm pretrained download in CI
    vit = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0, global_pool='')

    # Best-effort load of custom weights; if blocked, still return a valid model
    _maybe_load_custom_weights(vit)

    pre = functools.partial(load_preprocess_images, image_size=224)
    w = PytorchWrapper(identifier='vitpose_s_backbone', model=vit, preprocessing=pre)
    w.image_size = 224
    return w

def get_layers():
    return ['blocks.2', 'blocks.6', 'blocks.10', 'norm']
