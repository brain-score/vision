from .vjepa_model import VJEPA
import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.activations.temporal.utils import download_weight_file

from torchvision import transforms

class VJEPAWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.to(self._device)
        return self._model(tensor)  # encoder only

# Define the ImageNet mean and std
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

transform_img = transforms.Compose([
    transforms.Resize((224, 224)),  # Example size for ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

def transform_video(video):
    frames = []
    for img in video.to_pil_imgs():
        frames += [transform_img(img)]
    frames = th.stack(frames)
    return frames

def get_model(identifier, num_frames=16):
    assert identifier.startswith("VJEPA")

    if identifier == "VJEPA-L16":
        url = "https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar"
        H = W = 14
        vit_type = "vit_large"
        num_blocks = 24
    elif identifier == "VJEPA-H16":
        url = "https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar"
        H = W = 14
        vit_type = "vit_huge"
        num_blocks = 32
    elif identifier == "VJEPA-H16-384":
        url = "https://dl.fbaipublicfiles.com/jepa/vith16-384/vith16-384.pth.tar"
        H = W = 24
        vit_type = "vit_huge"
        num_blocks = 32
    else:
        raise ValueError(f"Unknown VJEPA identifier: {identifier}")

    T = None
    weight_path = download_weight_file(url, folder="temporal_model_vjepa")

    # Instantiate the model
    net = VJEPA(weight_path, vit_type)

    def process_output(layer, layer_name, inputs, output):
        if layer_name == "encoder.encoder.patch_embed":
            global T
            T = inputs[0].shape[2]

        B, L, C = output.shape
        assert L == T//2 * H * W
        output = output.view(B, T//2, H, W, C)

        return output

    inferencer_kwargs = {
        "fps": 10,
        "layer_activation_format": {
            "encoder.encoder.patch_embed": "THWC",
            **{f"encoder.encoder.blocks.{i}": "THWC" for i in range(0, num_blocks, 2)},
        },
        "duration": None,
        "time_alignment": "evenly_spaced",
        "process_output": process_output,
    }

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."

    wrapper = VJEPAWrapper(identifier, net, transform_video, 
                                **inferencer_kwargs)
    return wrapper
