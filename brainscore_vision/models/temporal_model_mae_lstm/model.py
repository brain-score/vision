import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file
from .mae_model import pfMAE_LSTM_physion, load_model

from torchvision import transforms as T

class MAELSTMWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.permute(0, 2, 1, 3, 4)
        tensor = tensor.to(self._device)
        with th.no_grad():
            output = self._model(tensor)
        return output#features  # encoder only

transform_img = T.Compose([T.Resize(256),
    T.CenterCrop(224)]) # ToTensor() divides by 255

def transform_video(video):
    frames = th.Tensor(video.to_numpy()).permute(0, 3, 1, 2)
    frames = transform_img(frames)
    return frames.permute(1, 0, 2, 3)

def get_model(identifier, num_frames=7):
    assert identifier.startswith("MAE-LSTM")
    
    model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/mae_lstm.pt", 
            version_id="Bf6kKIbSp5wQlWpXcZhVnSCsB7vgKDPv",
            sha1="0246e67f74dbfb72f272250252b32f20fd3e698f"
        )
    # Instantiate the model
    
    net = pfMAE_LSTM_physion(n_past=num_frames)
    net = load_model(net, model_path)

    if identifier == "MAE-LSTM-SIM":
        inferencer_kwargs = {
            "fps": 16,
            "layer_activation_format": {
                "dynamics": "TC",
            },
            "duration": None,
            "time_alignment": "evenly_spaced",
            "convert_img_to_video":True,
            "img_duration":450
        }
        
        def process_activation(layer, layer_name, inputs, output):
            return output["simulated_rollout_states"]
            
    elif identifier == "MAE-LSTM-SIM-OBSERVED":
        inferencer_kwargs = {
            "fps": 16,
            "layer_activation_format": {
                "dynamics": "TC",
            },
            "duration": None,
            "time_alignment": "evenly_spaced",
            "convert_img_to_video":True,
            "img_duration":450
        }
        
        def process_activation(layer, layer_name, inputs, output):
            return output["observed_dynamic_states"]
            
    elif identifier == "MAE-LSTM-ENCODER":
        inferencer_kwargs = {
            "fps": 16,
            "layer_activation_format": {
                "encoder": "TC",
            },
            "duration": None,
            "time_alignment": "evenly_spaced",
            "convert_img_to_video":True,
            "img_duration":450
        }
        
        def process_activation(layer, layer_name, inputs, output):
            return output["observed_encoder_states"]

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported." 

    wrapper = MAELSTMWrapper(identifier, net, transform_video, 
                                process_output=process_activation,
                                **inferencer_kwargs)

    return wrapper