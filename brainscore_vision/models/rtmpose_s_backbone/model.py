import functools, torch
from torch.hub import load_state_dict_from_url
from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper, load_preprocess_images
)
from mmdet.models.backbones.cspnext import CSPNeXt  # import from MMDetection

RTMPOSE_S_URL = "https://github.com/hs540/brainscore-pose-weights/releases/download/models/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.pth"

def _filter_to_backbone(sd):
    sd = sd.get("state_dict", sd)
    filtered = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    return filtered or sd

def get_model():
    backbone = CSPNeXt(arch='P5', deepen_factor=0.33, widen_factor=0.5, out_indices=(0,1,2,3,4))
    sd = load_state_dict_from_url(RTMPOSE_S_URL, map_location='cpu', check_hash=False)
    backbone.load_state_dict(_filter_to_backbone(sd), strict=False)
    pre = functools.partial(load_preprocess_images, image_size=256)
    w = PytorchWrapper(identifier='rtmpose_s_backbone', model=backbone, preprocessing=pre)
    w.image_size = 256
    return w

def get_layers():
    """
    Resolve robust layer taps across MMDet CSPNeXt variants.
    Tries in order: `stages.*`, then `stage1/2/3`, then falls back to best guesses.
    """
    from mmdet.models.backbones.cspnext import CSPNeXt
    # build a lightweight instance just to inspect names (no weights)
    probe = CSPNeXt(arch='P5', deepen_factor=0.33, widen_factor=0.5, out_indices=(0,1,2,3,4))

    taps = ['stem']

    # 1) Preferred: ModuleList `stages`
    if hasattr(probe, 'stages'):
        try:
            n = len(probe.stages)
            # choose three mid/late stages if available
            idxs = [i for i in range(n) if i > 0][:3] or list(range(min(3, n)))
            taps += [f'stages.{i}' for i in idxs]
            return taps
        except Exception:
            pass

    # 2) Named attributes `stage1`, `stage2`, `stage3`, …
    stage_attrs = [f'stage{i}' for i in range(1, 6) if hasattr(probe, f'stage{i}')]
    if stage_attrs:
        taps += stage_attrs[:3]
        return taps

    # 3) Fallback: pick top-level children with "stage" in the name
    stage_like = [name for name, _ in probe.named_children() if 'stage' in name.lower()]
    if stage_like:
        taps += stage_like[:3]
        return taps

    # 4) Last resort: pick some deeper blocks after the stem
    flat = [name for name, _ in probe.named_modules()]
    # find 3 non-root, non-stem modules
    candidates = [n for n in flat if n and not n.startswith('stem')][:3]
    return taps + candidates