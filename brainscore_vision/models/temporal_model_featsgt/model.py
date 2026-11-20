from .featsgt_model import FeatsModel
import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file

from torchvision import transforms


class FEATSGTWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.to(self._device)
        return self._model(tensor)  # encoder only

# Define the ImageNet mean and std
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
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
    assert identifier.startswith("FEATSGT")

    model_name = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/raft-sintel.pth", # change this
            version_id="QxI3fboAhgv82BnonbnM3cTouBvdL4N7",
            sha1="88b0c4569f7098e2921846b2cc8eb5af2e4db0fc"
        )
    
    # Instantiate the model

    net = FeatsModel(model_name)
    # probably need to load weights here

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
    
    wrapper = FEATSGTWrapper(identifier, net, transform_video, 
                                **inferencer_kwargs)
    return wrapper
