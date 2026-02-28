import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file
from .pn_model import pfPN_LSTM_physion, load_model

from torchvision import transforms as T

class PNLSTMWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.to(self._device)
        output = self._model(tensor)
        return output

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))
        
        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor

transform_img = T.Compose([T.Resize(64),
                           T.ToTensor(),
                           GroupNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def transform_video(video):
    frames = []
    for img in video.to_pil_imgs():
        frames += [transform_img(img)]
    frames = th.stack(frames)
    return frames

def get_model(identifier, num_frames=7):
    assert identifier.startswith("PN-LSTM")
    
    model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/pixelnerf_lstm.pt", 
            version_id="YV9zlBxLV4lBfVsAs1vIyh_gDkf7mu_j",
            sha1="14be113cd6f0ce4b9348cccc2018eb792ab2d0c1"
        )
    
    config_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/merged_conf.conf", 
            version_id="jjLOH81TcgEW9QBw8ZuuU.pjCMsXWwae",
            sha1="c83621086331949a05524b055e278a09c94fc43e"
        )
    # Instantiate the model
    net = pfPN_LSTM_physion(config_path, n_past=num_frames)
    net = load_model(net, model_path)

    if identifier == "PN-LSTM-SIM":
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
            
    elif identifier == "PN-LSTM-SIM-OBSERVED":
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
            
    elif identifier == "PN-LSTM-ENCODER":
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

    wrapper = PNLSTMWrapper(identifier, net, transform_video, 
                                process_output=process_activation,
                                **inferencer_kwargs)

    return wrapper
