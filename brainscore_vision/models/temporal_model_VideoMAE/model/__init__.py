from timm import create_model
from .run_videomae_vis import *  # must do
import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.models.temporal_models.weights import weight_registry   
from torchvision import transforms


class VideoMAEv1Wrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.to(self._device)
        return self._model.forward_encoder(tensor, mask=None)  # encoder only
    

input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
input_std = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=input_mean, std=input_std),
])

def transform_video(video):
    import torch
    frames = torch.Tensor(video.to_numpy() / 255.0).permute(0, 3, 1, 2)
    frames = transform_img(frames)
    return frames.permute(1, 0, 2, 3)


def get_model(identifier, num_frames=16):
    assert identifier.startswith("VideoMAE-V1")
    pretrain_only = True

    if identifier == "VideoMAE-V1-B":
        model_name = "pretrain_videomae_base_patch16_224"
        pth = weight_registry['VideoMAE/vit_b_k400_pt_1200e.pth']
        num_blocks = 12
    elif identifier == "VideoMAE-V1-L":
        model_name = "pretrain_videomae_large_patch16_224"
        pth = weight_registry['VideoMAE/vit_l_k400_pt_1200e.pth']
        num_blocks = 24

    # Instantiate the model

    net = create_model(
        model_name,
        pretrained=False,
    )

    # Load the model weights
    if pretrain_only:
        st = th.load(pth, map_location='cpu')['model']
        msg = net.load_state_dict(st, strict=False)  # encoder only
    else:
        pth = weight_registry['VideoMAE/videomae-v1-vit-b-k400-finetune.pth']
        st_ = th.load(pth, map_location='cpu')['module']
        st = {}
        for k, v in st_.items():
            st['encoder.'+k] = v
        msg = net.load_state_dict(st, strict=False) 
    for layer in msg.missing_keys:
        assert layer.startswith("decoder.")

    feature_map_size = 14

    inferencer_kwargs = {
        "fps": 6.25,
        "layer_activation_format": {
            "encoder.patch_embed": "THWC",
            **{f"encoder.blocks.{i}": "THWC" for i in range(num_blocks)},
        },
        "num_frames": num_frames,
    }

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."

    def process_activation(layer, layer_name, inputs, output):
        B = output.shape[0]
        C = output.shape[-1]
        output = output.reshape(B, -1, feature_map_size, feature_map_size, C)
        return output

    wrapper = VideoMAEv1Wrapper(identifier, net, transform_video, 
                                process_output=process_activation,
                                **inferencer_kwargs)
    return wrapper
