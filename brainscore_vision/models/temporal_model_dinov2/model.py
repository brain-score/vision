from .dinov2_model import pfDINOV2
import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from torchvision import transforms


class DINOV2Wrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.to(self._device)
        tensor = self._model(tensor)  # encoder only
        return tensor

# Define the ImageNet mean and std
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

transform_img = transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224)
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),])

def transform_video(video):
    frames = []
    for img in video.to_pil_imgs():
        frames += [transform_img(img)]
    frames = th.stack(frames)
    return frames


def get_model(identifier, num_frames=16):
    assert identifier.startswith("DINO")
    
    if identifier == "DINO-GIANT-Temporal":
        model_name = "facebook/dinov2-giant"
    if identifier == "DINO-LARGE-Temporal":
        model_name = "facebook/dinov2-large"
    elif identifier == "DINO-BASE-Temporal":
        model_name = "facebook/dinov2-base"

    # Instantiate the model

    net = pfDINOV2(model_name)

    inferencer_kwargs = {
        "fps": 10,
        "layer_activation_format": {
            "encoder": "TC",
        },
        "duration": None,#(0, 450),
        "time_alignment": "evenly_spaced",
        "convert_img_to_video":True,
        "img_duration":450
    }

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."
    
    wrapper = DINOV2Wrapper(identifier, net, transform_video, 
                                **inferencer_kwargs)
    return wrapper
