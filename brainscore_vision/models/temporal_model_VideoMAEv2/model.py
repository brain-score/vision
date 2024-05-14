
import numpy as np
import torch
from timm.models import create_model
from torchvision import transforms

# NOTE: Do not comment `import models`, it is used to register models
from videomae_v2 import *  # noqa: F401

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file


LAYER_SELECT_STEP = 2

def to_normalized_float_tensor(vid):
    vid = torch.Tensor(vid.to_numpy())
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255

# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)

class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)
    

transform_video = transforms.Compose(
    [ToFloatTensorInZeroOne(),
        Resize((224, 224))])

def get_model(identifier):

    if identifier == "VideoMAE-V2-G":
        model_name = "vit_giant_patch14_224"
        pth = load_weight_file(
            bucket="brainscore-vision",
            relative_path="temporal_model_VideoMAEv2/vit_g_hybrid_pt_1200e.pth",
            version_id="TxtkfbeMV105dzpzTwi0Kn5glnvQvIrq",
            # sha1="9048f2bc0b0c7ba4d0e5228f3a7c0bef4dbaca69",
            sha1="32126231526fe310a6aba20c16d0e6435f5f0bb8"
        )
        num_blocks = 40
        feature_map_size = 16
    elif identifier == "VideoMAE-V2-B":
        model_name = "vit_base_patch16_224"
        pth = load_weight_file(
            bucket="brainscore-vision",
            relative_path="temporal_model_VideoMAEv2/vit_b_hybrid_pt_800e.pth",
            version_id="rRjpYq21dAQ5KaCLbEHK.YaLZ_fbMPKw",
            sha1="1e3602691964b1eb6f7c33529119243a5b235635"
        )
        num_blocks = 12
        feature_map_size = 14
        
    num_frames = 16

    model = create_model(model_name)
    
    ckpt = torch.load(pth, map_location='cpu')
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break

    encoder_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("encoder."):
            encoder_ckpt[k[8:]] = v

    msg = model.load_state_dict(encoder_ckpt, strict=False)
    print(msg)

    inferencer_kwargs = {
        "fps": 6.25,
        "layer_activation_format": {
            "patch_embed": "THWC",
            **{f"blocks.{i}": "THWC" for i in range(0, num_blocks, LAYER_SELECT_STEP)},
            # "head": "THWC"  # weight not available
        },
        "num_frames": num_frames,
    }

    def process_activation(layer, layer_name, inputs, output):
        B = output.shape[0]
        C = output.shape[-1]
        output = output.reshape(B, -1, feature_map_size, feature_map_size, C)
        return output

    wrapper = PytorchWrapper(identifier, model, transform_video, 
                                process_output=process_activation,
                                **inferencer_kwargs)
    return wrapper
