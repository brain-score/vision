import torch as th
import numpy as np

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file
from .dino_model import pfDINO_LSTM_physion, load_model

from torchvision import transforms as T

class DINOLSTMWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.to(self._device)
        output = self._model(tensor)
        return output#features  # encoder only

# Define the ImageNet mean and std
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

transform_img = T.Compose([
    T.Resize(512),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),])

def transform_video(video):
    frames = []
    for img in video.to_pil_imgs():
        frames += [transform_img(img)]
    frames = th.stack(frames)
    return frames

def get_model(identifier, num_frames=7):
    assert identifier.startswith("DINO-LSTM")

    model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/dino_lstm.pt", 
            version_id="EuKBPN7lh_WSE6uyR2Gs0.thPRo51tL6",
            sha1="229c0f9eb4039c698039e843c07e76c67ebfde44"
        )

    net = pfDINO_LSTM_physion(n_past=num_frames)
    net = load_model(net, model_path)

    if identifier == "DINO-LSTM-SIM":
        inferencer_kwargs = {
            "fps": 16,
            "batch_size": 2,
            "layer_activation_format": {
                "dynamics": "TC",
            },
            "duration": None,
            "time_alignment": "evenly_spaced",
        }
        
        def process_activation(layer, layer_name, inputs, output):
            return output["simulated_rollout_states"]
            
    elif identifier == "DINO-LSTM-SIM-OBSERVED":
        inferencer_kwargs = {
            "fps": 16,
            "batch_size": 2,
            "layer_activation_format": {
                "dynamics": "TC",
            },
            "duration": None,
            "time_alignment": "evenly_spaced",
        }
        
        def process_activation(layer, layer_name, inputs, output):
            return output["observed_dynamic_states"]
            
    elif identifier == "DINO-LSTM-ENCODER":
        inferencer_kwargs = {
            "fps": 16,
            "batch_size": 2,
            "layer_activation_format": {
                "encoder": "TC",
            },
            "duration": None,
            "time_alignment": "evenly_spaced",
        }
        
        def process_activation(layer, layer_name, inputs, output):
            return output["observed_encoder_states"]

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."

    wrapper = DINOLSTMWrapper(identifier, net, transform_video,
                                process_output=process_activation,
                                **inferencer_kwargs)

    return wrapper