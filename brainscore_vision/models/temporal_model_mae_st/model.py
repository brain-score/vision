import torch
from iopath.common.file_io import g_pathmgr as pathmgr
from mae_st import models_vit
from mae_st.models_vit import VisionTransformer, nn, partial
from mae_st.util import misc 
from mae_st.util.pos_embed import interpolate_pos_embed

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file


def vit_huge_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

models_vit.__dict__["vit_huge_patch16"] = vit_huge_patch16


LAYER_SELECT_STEP = 2
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
        num_blocks = 24
        feature_map_size = 14
        load_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="temporal_model_mae_st/mae_pretrain_vit_large_k400.pth", 
            version_id="cPcP4AzpG95CimQ5Pn.CHKnGUJlLXM3m",
            sha1="c7fb91864a4ddf8b99309440121a3abe66b846bb"
        )

    elif identifier == "MAE-ST-G":
        model_name = "vit_huge_patch16"
        num_blocks = 32
        feature_map_size = 14
        load_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="temporal_model_mae_st/mae_pretrain_vit_huge_k400.pth", 
            version_id="IYKa8QiocgBzo3EhsBouS62HboK6iqYT",
            sha1="177e48577142ca01949c08254834ffa1198b9eb4"
        )

    num_frames = 16
    t_patch_size = 2

    model = models_vit.__dict__[model_name](
        num_frames=num_frames,
        t_patch_size=t_patch_size,
        cls_embed=True,
        sep_pos_embed=True
    )

    with pathmgr.open(load_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    print("Load pre-trained checkpoint from: %s" % load_path)
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
            **{f"blocks.{i}": "THWC" for i in range(0, num_blocks, LAYER_SELECT_STEP)},
            # "head": "THWC"  # weight not available
        },
        "num_frames": num_frames,
    }

    def process_activation(layer, layer_name, inputs, output):
        B = output.shape[0]
        C = output.shape[-1]
        if layer_name.startswith("blocks"):
            output = output[:, 1:]  # remove cls token
        output = output.reshape(B, -1, feature_map_size, feature_map_size, C)
        return output

    wrapper = PytorchWrapper(identifier, model, transform_video, 
                                process_output=process_activation,
                                **inferencer_kwargs)

    return wrapper
