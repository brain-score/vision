from torch.hub import load_state_dict_from_url
from pathlib import Path
import functools, torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images

# 👉 replace with your actual GitHub Release asset URL
RTMPOSE_S_URL = "https://github.com/hs540/brainscore-pose-weights/releases/download/models/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.pth"

def _filter_to_backbone(sd):
    sd = sd.get("state_dict", sd)
    filtered = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            filtered[k[len("backbone."):]] = v
    return filtered or sd  # fallback if already backbone-only

def get_model():
    from mmpose.models.backbones.cspnext import CSPNeXt  # CSPNeXt = RTMPose backbone
    backbone = CSPNeXt(arch="P5", deepen_factor=0.33, widen_factor=0.5, out_indices=(0,1,2,3,4))

    # download weights at runtime from your release
    sd = load_state_dict_from_url(RTMPOSE_S_URL, map_location="cpu", check_hash=False)
    sd = _filter_to_backbone(sd)
    backbone.load_state_dict(sd, strict=False)

    pre = functools.partial(load_preprocess_images, image_size=256)
    w = PytorchWrapper(identifier="rtmpose_s_backbone", model=backbone, preprocessing=pre)
    w.image_size = 256
    return w

def get_layers():
    return ["stem", "stages.1", "stages.2", "stages.3"]

def get_bibtex():
    return r"""@article{jiang2023rtmpose, title={RTMPose: Real-Time Multi-Person Pose Estimation}, journal={arXiv:2303.07399}, year={2023}}"""
