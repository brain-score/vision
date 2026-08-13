import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file
from .r3m_model import pfR3M_LSTM_physion, load_model

from torchvision import transforms as T

class R3MLSTMWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.to(self._device)
        output = self._model(tensor)
        return output

transform_img = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()])

def transform_video(video):
    frames = []
    for img in video.to_pil_imgs():
        frames += [transform_img(img)]
    frames = th.stack(frames)
    return frames

def get_model(identifier, num_frames=7):
    assert identifier.startswith("R3M-LSTM")
    pretrain_only = True

    if identifier.startswith("R3M-LSTM-EGO4D"):
        model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/r3m_lstm_ego4d.pt", 
            version_id="Fz1LBUmhex5tT7tRuxC2ThLtzFUwSIx0",
            sha1="8ae184cdfff3014b3dcdfffe9d52457a66339d32"
        )
    elif identifier.startswith("R3M-LSTM-PHYS"):
        model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/r3m_lstm_physion.pt", 
            version_id="2ugg.9Hfp1s9E4MWvtfKZL6onawFHxSy",
            sha1="b09b87a965a0ef5086224ce474af20e79a34d595"
        )

    # Instantiate the model
        
    net = pfR3M_LSTM_physion(n_past=num_frames)
    net = load_model(net, identifier, model_path)

    if identifier == "R3M-LSTM-EGO4D-SIM" or identifier == "R3M-LSTM-PHYS-SIM":
        inferencer_kwargs = {
            "fps": 16,
            "layer_activation_format": {
                "dynamics": "TC",
            },
            "duration": None,
            "time_alignment": "evenly_spaced",
        }
        
        def process_activation(layer, layer_name, inputs, output):
            return output["simulated_rollout_states"]
            
    elif identifier == "R3M-LSTM-EGO4D-SIM-OBSERVED" or identifier == "R3M-LSTM-PHYS-SIM-OBSERVED":
        inferencer_kwargs = {
            "fps": 16,
            "layer_activation_format": {
                "dynamics": "TC",
            },
            "duration": None,
            "time_alignment": "evenly_spaced",
        }
        
        def process_activation(layer, layer_name, inputs, output):
            return output["observed_dynamic_states"]
            
    if identifier == "R3M-LSTM-EGO4D-ENCODER" or identifier == "R3M-LSTM-PHYS-ENCODER":
        inferencer_kwargs = {
            "fps": 16,
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

    wrapper = R3MLSTMWrapper(identifier, net, transform_video, 
                                process_output=process_activation,
                                **inferencer_kwargs)
    return wrapper
