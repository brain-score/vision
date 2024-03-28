import torch

from gdt_model.model import GDT
from gdt_model.video_transforms import clip_augmentation

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file


def transform_video(video):
    arr = video.to_numpy()
    arr = torch.as_tensor(arr)
    return clip_augmentation(arr)


def get_model(identifier):

    assert identifier.startswith("GDT-")
    dataset = "-".join(identifier.split("-")[1:])

    if dataset == "Kinetics400":
        pth = load_weight_file(
            bucket="brainscore-vision",
            relative_path="temporal_model_GDT/gdt_K400.pth",
            version_id="JpU_tnCzrbTejn6sOrQMk8eRsJ97yFgt",
            sha1="7f12c60670346b1aab15194eb44c341906e1bca6"
        )
    elif dataset == "IG65M":
        pth = load_weight_file(
            bucket="brainscore-vision",
            relative_path="temporal_model_GDT/gdt_IG65M.pth",
            version_id="R.NoD6VAbFbJdf8tg5jnXIWB3hQ8GlSD",
            sha1="3dcee3af61691e1e7e47e4b115be6808f4ea8172"
        )
    elif dataset == "HowTo100M":
        pth = load_weight_file(
            bucket="brainscore-vision",
            relative_path="temporal_model_GDT/gdt_HT100M.pth",
            version_id="BVRl9t_134PoKZCn9W54cyfkImCW2ioq",
            sha1="a9a979c82e83b955794814923af736eb34e6f080"
        )
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