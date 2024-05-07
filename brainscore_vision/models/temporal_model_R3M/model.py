from r3m import load_r3m
import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
#from brainscore_vision.model_helpers.s3 import load_weight_file
from torchvision import transforms


class R3MWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.to(self._device)
        return self._get_encoder_feats(tensor)  # encoder only
    
    def _get_encoder_feats(self, x):
        # applies encoder to each image in x: (Bs, T, 3, H, W) or (Bs, 3, H, W)
        with torch.no_grad():
            feats = []
            for _x in torch.split(x, 1, dim=1):
                _x = torch.squeeze(
                        _x, dim=1
                ) 
                feats.append(self._extract_feats(_x))
        return torch.stack(feats, axis=1)

    def _extract_feats(self, x):
        feats = self._model(x)
        feats = torch.flatten(feats, start_dim=1)  # (Bs, -1)
        return feats

transform_img = transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]) # ToTensor() divides by 255

def transform_video(video):
    import torch
    frames = torch.Tensor(video.to_numpy()).permute(0, 3, 1, 2)
    frames = transform_img(frames)
    return frames.permute(1, 0, 2, 3)


def get_model(identifier, num_frames=16):
    assert identifier.startswith("R3M")
    pretrain_only = True

    if identifier == "R3M-ResNet50":
        model_name = "resnet50"
    elif identifier == "R3M-ResNet18":
        model_name = "resnet18"
    elif identifier == "R3M-ResNet34":
        model_name = "resnet34"

    # Instantiate the model

    net = load_r3m(model_name)

    num_blocks = 4
    inferencer_kwargs = {
        "fps": 100,
        "layer_activation_format": {
            "convnet": "TC",
            "convnet.conv1": "TCHW",
            **{f"convnet.layer{i}": "TCHW" for i in range(1, num_blocks)},
        },
        "duration": (0, 450),
    }

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."

    wrapper = R3MWrapper(identifier, net, transform_video, 
                                process_output=None,
                                **inferencer_kwargs)
    return wrapper
