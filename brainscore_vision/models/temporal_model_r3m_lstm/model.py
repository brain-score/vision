import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file
from .r3m_model import pfR3M_LSTM_physion, load_model

from torchvision import transforms as T

class R3MLSTMWrapper(PytorchWrapper):
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
    assert identifier.startswith("R3M-LSTM")
    pretrain_only = True

    if identifier == "R3M-LSTM-EGO4D":
        model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/r3m_lstm_ego4d.pt", 
            version_id="Fz1LBUmhex5tT7tRuxC2ThLtzFUwSIx0",
            sha1="8ae184cdfff3014b3dcdfffe9d52457a66339d32"
        )
    elif identifier == "R3M-LSTM-PHYS":
        model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/r3m_lstm_physion.pt", 
            version_id="2ugg.9Hfp1s9E4MWvtfKZL6onawFHxSy",
            sha1="b09b87a965a0ef5086224ce474af20e79a34d595"
        )

    # Instantiate the model
        
    net = pfR3M_LSTM_physion(n_past=num_frames)
    net = load_model(net, identifier, model_path)

    inferencer_kwargs = {
        "fps": 16,
        "layer_activation_format": {
            "encoder": "TC",
            "dynamics": "TC",
        },
        "duration": None,
        "time_alignment": "evenly_spaced",
        "convert_img_to_video":True,
        "img_duration":450
    }

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."

    def process_activation(layer, layer_name, inputs, output):
        if layer_name == 'encoder':
            activations = output["observed_states"]
        else:
            activations = output["rollout_states"]
        return activations 

    wrapper = R3MLSTMWrapper(identifier, net, transform_video, 
                                process_output=process_activation,
                                **inferencer_kwargs)
    return wrapper
