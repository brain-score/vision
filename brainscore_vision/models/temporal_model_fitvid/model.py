import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file
from .fitvid_model import FitVidEncoder, load_model

from torchvision import transforms as T

class FitVidWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.permute(0, 2, 1, 3, 4)
        tensor = tensor.to(self._device)
        with th.no_grad():
            output = self._model(tensor)
        return output#features  # encoder only

transform_img = T.Compose([T.Resize(64)])

def transform_video(video):
    frames = th.Tensor(video.to_numpy()).permute(0, 3, 1, 2)
    frames = transform_img(frames)
    return frames.permute(1, 0, 2, 3)


def get_model(identifier, num_frames=7):
    assert identifier.startswith("FITVID")
    pretrain_only = True

    if identifier == "FITVID-EGO4D-OBSERVED":
        model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/fitvid_ego4d.pt", 
            version_id="1cXzv4b9cPlnhSQU4zzmeRBgEdio9VFw",
            sha1="1764e964abc51d0b06e27cef46cfa4702391f3cc"
        )
    elif identifier == "FITVID-PHYS-OBSERVED":
        model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/fitvid_physion.pt", 
            version_id="inezcqO81.4Kpuouzba3sxEZiw5Bgoig",
            sha1="7a2fa16d235337182b08db3c467d66ae3e0c9333"
        )
    
    net = FitVidEncoder(n_past=num_frames)
    net = load_model(net, model_path)

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

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."

    def process_activation(layer, layer_name, inputs, output):
        activations = output["h_preds"]
        return activations 

    wrapper = FitVidWrapper(identifier, net, transform_video, 
                                process_output=process_activation,
                                **inferencer_kwargs)
    return wrapper