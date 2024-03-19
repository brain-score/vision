import torch

from .model import load_model
from .video_transforms import clip_augmentation

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper


def transform_video(video):
    arr = video.to_numpy()
    arr = torch.as_tensor(arr)
    return clip_augmentation(arr)


def get_model(identifier):

    assert identifier.startswith("SeLaVi-")
    dataset = "-".join(identifier.split("-")[1:])

    if dataset == "Kinetics400":
        pth = "/home/ytang/workspace/data/weights/temporal_model_SeLaVi/selavi_kinetics.pth"
        num_classes = 400
    elif dataset == "Kinetics-Sound":
        pth = "/home/ytang/workspace/data/weights/temporal_model_SeLaVi/selavi_kinetics_sound.pth"
        num_classes = 32
    elif dataset == "VGG-Sound":
        pth = "/home/ytang/workspace/data/weights/temporal_model_SeLaVi/selavi_vgg_sound.pth"
        num_classes = 309
    elif dataset == "AVE":
        pth = "/home/ytang/workspace/data/weights/temporal_model_SeLaVi/selavi_ave.pth"
        num_classes = 28
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load model
    model = load_model(
        vid_base_arch="r2plus1d_18",
        aud_base_arch="resnet9",
        use_mlp=True,
        num_classes=num_classes,
        pretrained=False,
        norm_feat=False,
        use_max_pool=False,
        headcount=10,
    )

    model = model.video_network  # Remove audio network

    # Load weights
    state_dict_ = torch.load(pth, map_location="cpu")['model']
    state_dict = {}
    for k, v in list(state_dict_.items()):
        if k.startswith("module.video_network."):
            k = k[len("module.video_network."):]
            state_dict[k] = v
    model.load_state_dict(state_dict)

    layer_activation_format = {
        "base.stem": "CTHW",
        **{f"base.layer{i}": "CTHW" for i in range(1, 5)},
        # "base.fc": "C",  # no fc
    }

    return PytorchWrapper(identifier, model, transform_video, fps=30, layer_activation_format=layer_activation_format)