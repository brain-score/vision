from torch.hub import load_state_dict_from_url
from pathlib import Path
import math, functools, torch, torch.nn.functional as F
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images

# 👉 replace with your actual GitHub Release asset URL
VITPOSE_S_URL = "https://github.com/hs540/brainscore-pose-weights/releases/download/models/vitpose_small.pth"

def _interpolate_pos_embed(state_dict, model):
    if "pos_embed" not in state_dict:  # ok if checkpoint doesn’t store it
        return state_dict
    pe = state_dict["pos_embed"]           # [1, N+1, C]
    cls_tok, grid = pe[:, :1], pe[:, 1:]   # split cls + patch grid
    n = grid.shape[1]
    patch = getattr(model, "patch_size", 16)
    H, W = (256, 192)                      # typical ViTPose-S train size
    gh_new, gw_new = H // patch, W // patch
    g_old = int(math.sqrt(n))
    if g_old * g_old == n: gh_old, gw_old = g_old, g_old
    else:
        # rectangular fallback
        candidates = [(i, n // i) for i in range(1, n+1) if n % i == 0]
        gh_old, gw_old = min(candidates, key=lambda t: abs(t[0]-gh_new)+abs(t[1]-gw_new))
    grid = grid.reshape(1, gh_old, gw_old, -1).permute(0,3,1,2)
    grid = F.interpolate(grid, size=(gh_new, gw_new), mode="bicubic", align_corners=False)
    grid = grid.permute(0,2,3,1).reshape(1, gh_new*gw_new, -1)
    state_dict["pos_embed"] = torch.cat([cls_tok, grid], dim=1)
    return state_dict

def _filter_to_backbone(sd):
    sd = sd.get("state_dict", sd)
    filtered = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    return filtered or sd

def get_model():
    # ViTPose uses a **plain, non-hierarchical ViT backbone** + light decoder; we only wrap the backbone
    from mmpose.models.backbones.vit import ViT
    vit = ViT(arch="small", img_size=(256, 192), patch_size=16, final_norm=True, with_cls_token=True, drop_path_rate=0.1)

    sd = load_state_dict_from_url(VITPOSE_S_URL, map_location="cpu", check_hash=False)
    sd = _filter_to_backbone(sd)
    sd = _interpolate_pos_embed(sd, vit)   # handle 256×192 vs 224×224 grids
    vit.load_state_dict(sd, strict=False)

    pre = functools.partial(load_preprocess_images, image_size=256)
    w = PytorchWrapper(identifier="vitpose_s_backbone", model=vit, preprocessing=pre)
    w.image_size = 256
    return w

def get_layers(): return ["blocks.2", "blocks.6", "blocks.10", "norm"]

def get_bibtex():
    return r"""@article{xu2022vitpose, title={{ViTPose}: Simple Vision Transformer Baselines for Human Pose Estimation}, journal={arXiv:2204.12484}, year={2022}}"""
