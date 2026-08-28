import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file
from .pn_model import pfPN

from torchvision import transforms as T

class PNWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.permute(0, 2, 1, 3, 4)
        tensor = tensor.to(self._device)
        with th.no_grad():
            output = self._model(tensor)
        return output#features  # encoder only

transform_img = T.Compose([T.Resize(256),
    T.CenterCrop(224)])

def transform_video(video):
    frames = th.Tensor(video.to_numpy()).permute(0, 3, 1, 2)
    frames = transform_img(frames)
    return frames.permute(1, 0, 2, 3)

def get_model(identifier, num_frames=7):
    assert identifier.startswith("PixelNerf-Temporal")
    
    model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/pixel_nerf_latest", 
            version_id="No3poAbMlHORaazW3sw8GC0Adlp1tdJS",
            sha1="4c7338cbd8cb9cc6ed94e8dfde40bed1096e3a73"
        )

    config_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/merged_conf.conf", 
            version_id="jjLOH81TcgEW9QBw8ZuuU.pjCMsXWwae",
            sha1="c83621086331949a05524b055e278a09c94fc43e"
        )
    # Instantiate the model
    net = pfPN(model_path, config_path)
    
    inferencer_kwargs = {
        "fps": 16,
        "layer_activation_format": {
            "encoder": "TC",
        },
        "duration": None,
        "time_alignment": "evenly_spaced",
        "convert_img_to_video":True,
        "img_duration":900
    }

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."

    wrapper = PNWrapper(identifier, net, transform_video, 
                                **inferencer_kwargs)

    return wrapper