import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file
from .mcvd_model import MCVDSimulator

from torchvision import transforms as T

class MCVDWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.permute(0, 2, 1, 3, 4)
        tensor = tensor.to(self._device)
        with th.no_grad():
            output = self._model(tensor)
        return output#features  # encoder only

transform_img = T.Compose([T.Resize(64)]) # ToTensor() divides by 255

def transform_video(video):
    frames = th.Tensor(video.to_numpy()).permute(0, 3, 1, 2)
    frames = transform_img(frames)
    return frames.permute(1, 0, 2, 3)


def get_model(identifier, num_frames=7):
    assert identifier.startswith("MCVD")
    pretrain_only = True

    if identifier == "MCVD-EGO4D-SIM":
        model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/mcvd_ego4d.pt", 
            version_id="7_2SZtf5kXqmmTsJOPmkABI._HAX519c",
            sha1="ead964db02a855672b97f7a0b6d6c43c6b20ec88"
        )
        config_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/config_ego4d.yml", 
            version_id="qzKSMsJLuj8al1F.lOUxjvkYMzGP09il",
            sha1="00fc95dda440ceeb780878f0f0ab0f4ec9b16359"
        )
    elif identifier == "MCVD-PHYS-SIM":
        model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/mcvd_physion.pt", 
            version_id="Uo86X1URRUoqSAEOJ5oPhrt_bCfxb4ho",
            sha1="0a757d4b6693d2c5890b0ea909ca4aaedc76453c"
        )
        config_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="neuroai_stanford_weights/config_physion.yml", 
            version_id="p6ilLbUYCuhT5dBE97QkrA0GPQTlXLC6",
            sha1="ba66f9f5b37db0049bb68f68279fe060a3b3f89a"
        )
        
    # Instantiate the model
    
    net = MCVDSimulator(model_path, config_path)

    inferencer_kwargs = {
        "fps": 25,
        "layer_activation_format": {
            "dynamics": "TCHW",
        },
        "duration": None,
        "time_alignment": "evenly_spaced",
        "convert_img_to_video":True,
        "img_duration":450
    }

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."

    wrapper = MCVDWrapper(identifier, net, transform_video, 
                                **inferencer_kwargs)
    return wrapper
