import torch

from .model import GDT
from .video_transforms import clip_augmentation

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper


def transform_video(video):
    arr = video.to_numpy()
    arr = torch.as_tensor(arr)
    return clip_augmentation(arr)


def get_model(identifier):

    assert identifier.startswith("GDT-")
    dataset = "-".join(identifier.split("-")[1:])

    if dataset == "Kinetics400":
        pth = "/home/ytang/workspace/data/weights/temporal_model_GDT/gdt_K400.pth"
    elif dataset == "IG65M":
        pth = "/home/ytang/workspace/data/weights/temporal_model_GDT/gdt_IG65M.pth"
    elif dataset == "HowTo100M":
        pth = "/home/ytang/workspace/data/weights/temporal_model_GDT/gdt_HT100M.pth"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load model
    model = GDT(
        vid_base_arch="r2plus1d_18", 
        aud_base_arch="resnet9",
        pretrained=False, 
        norm_feat=False, 
        use_mlp=False,
        num_classes=256, 
    )

    model = model.video_network  # Remove audio network

    # Load weights
    state_dict_ = torch.load(pth, map_location="cpu")['model']
    state_dict = {}
    for k, v in list(state_dict_.items()):
        if k.startswith("video_network."):
            k = k[len("video_network."):]
            state_dict[k] = v
    model.load_state_dict(state_dict)

    layer_activation_format = {
        "base.stem": "CTHW",
        **{f"base.layer{i}": "CTHW" for i in range(1, 5)},
        "base.fc": "C",  # no fc
    }

    return PytorchWrapper(identifier, model, transform_video, fps=30, layer_activation_format=layer_activation_format)