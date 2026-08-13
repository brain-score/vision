import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file
from .resnet_model import pfResNet_LSTM_physion, load_model

from torchvision import transforms as T

class RESNETLSTMWrapper(PytorchWrapper):
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

transform_img = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    GroupNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def transform_video(video):
    frames = []
    for img in video.to_pil_imgs():
        frames += [transform_img(img)]
    frames = th.stack(frames)
    return frames

def get_model(identifier, num_frames=7):
    assert identifier.startswith("RESNET-LSTM")
    # Instantiate the model
    
    model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/resnet_lstm.pt", 
            version_id="F1VgIiJONrw.PDcaoxS_l2JjaM.8RIvG",
            sha1="41faf4b43c78045591e9c33e9670035c81ce6daa"
        )
    
    net = pfResNet_LSTM_physion(n_past=num_frames)
    net = load_model(net, model_path)

    if identifier == "RESNET-LSTM-SIM":
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
    elif identifier == "RESNET-LSTM-SIM-OBSERVED":
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
    elif identifier == "RESNET-LSTM-ENCODER":
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

    wrapper = RESNETLSTMWrapper(identifier, net, transform_video, 
                                process_output=process_activation,
                                **inferencer_kwargs)
    return wrapper
