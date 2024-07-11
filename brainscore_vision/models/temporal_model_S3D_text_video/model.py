import torch
import numpy as np
from torchvision import transforms
from s3dg_howto100m import S3D

from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file


img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
])

def transform_video(video):
    frames = video.to_numpy() / 255.
    frames = torch.Tensor(frames)
    frames = frames.permute(0, 3, 1, 2)
    frames = img_transform(frames)
    return frames.permute(1, 0, 2, 3)


def get_model(identifier="s3d-HowTo100M"):
    inferencer_kwargs = {
        "fps": 24,  # common YouTube frame rate
        "layer_activation_format": 
        {
            "conv1": "CTHW",
            "conv_2c": "CTHW",
            "mixed_3c": "CTHW",
            "mixed_4b": "CTHW",
            "mixed_4d": "CTHW",
            "mixed_4f": "CTHW",
            "mixed_5c": "CTHW",
            "fc": "C"
        },
    }
    process_output = None

    model_name = identifier

    model_pth = load_weight_file(
        bucket="brainscore-vision",
        relative_path="temporal_model_S3D_text_video/s3d_howto100m.pth",
        version_id="hRp6I8bpwreIMUVL0H.zCdK0hqRggL7n",
        sha1="31e99d2a1cd48f2259ca75e719ac82c8b751ea75"
    )

    dict_pth = load_weight_file(
        bucket="brainscore-vision",
        relative_path="temporal_model_S3D_text_video/s3d_dict.npy",
        version_id="4NxVLe8DSL6Uue0F7e2rz8HZuOk.tkBI",
        sha1="d368ff7d397ec8240f1f963b5efe8ff245bac35f"
    )

    # Instantiate the model
    model = S3D(dict_pth, 512)

    # Load the model weights
    model.load_state_dict(torch.load(model_pth))

    wrapper = PytorchWrapper(identifier, model, transform_video, 
                             process_output=process_output,
                             **inferencer_kwargs)
    
    return wrapper