from .mae_model import pfMAE
import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from torchvision import transforms


class MAEWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.to(self._device)
        return self._model(tensor)  # encoder only

# Define the ImageNet mean and std
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Define the transform
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),  # Example size for ViT
    transforms.ToTensor()
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

def transform_video(video):
    frames = []
    for img in video.to_pil_imgs():
        frames += [transform_img(img)]
    frames = th.stack(frames)
    return frames


def get_model(identifier, num_frames=16):
    assert identifier.startswith("MAE")

    if identifier == "MAE-BASE-Temporal":
        model_name = "facebook/vit-mae-base"
    elif identifier == "MAE-LARGE-Temporal":
        model_name = "facebook/vit-mae-large"

    # Instantiate the model

    net = pfMAE(model_name)

    inferencer_kwargs = {
        "fps": 10,
        "layer_activation_format": {
            "encoder": "TC",
        },
        "duration": None,
        "time_alignment": "evenly_spaced",
    }

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."
    
    wrapper = MAEWrapper(identifier, net, transform_video, 
                                **inferencer_kwargs)
    return wrapper
