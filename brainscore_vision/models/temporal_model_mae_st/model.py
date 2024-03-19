import sys

import torch
from .models import models_vit, pathmgr, interpolate_pos_embed, misc

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.models.temporal_models.weights import weight_registry   


mean = (0.45, 0.45, 0.45)
std = (0.225, 0.225, 0.225)

from torchvision import transforms

transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean, std),
])


def transform_video(video):
    import torch
    frames = torch.Tensor(video.to_numpy() / 255.0).permute(0, 3, 1, 2)
    frames = transform_img(frames)
    return frames.permute(1, 0, 2, 3)



def get_model(identifier):
    if identifier == "MAE-ST-L":
        model_name = "vit_large_patch16"
        pth = weight_registry["mae_st/mae_pretrain_vit_large_k400.pth"]
        num_blocks = 24
        feature_map_size = 14

    num_frames = 16
    t_patch_size = 2

    model = models_vit.__dict__[model_name](
        num_frames=num_frames,
        t_patch_size=t_patch_size
    )

    with pathmgr.open(pth, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    print("Load pre-trained checkpoint from: %s" % pth)
    if "model" in checkpoint.keys():
        checkpoint_model = checkpoint["model"]
    else:
        checkpoint_model = checkpoint["model_state"]
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    checkpoint_model = misc.convert_checkpoint(checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    inferencer_kwargs = {
        "fps": 6.25,
        "layer_activation_format": {
            "patch_embed": "THWC",
            **{f"blocks.{i}": "THWC" for i in range(num_blocks)},
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
