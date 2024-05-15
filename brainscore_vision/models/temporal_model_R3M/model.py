from r3m import load_r3m
import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
#from brainscore_vision.model_helpers.s3 import load_weight_file
from torchvision import transforms


class R3MWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.permute(0, 2, 1, 3, 4)
        tensor = tensor.to(self._device)
        r = self._get_encoder_feats(tensor)  # encoder only
        return r#.squeeze(1)

    def _get_encoder_feats(self, x):
        # applies encoder to each image in x: (Bs, T, 3, H, W) or (Bs, 3, H, W)
        with th.no_grad():
            feats = []
            for _x in th.split(x, 1, dim=1):
                _x = th.squeeze(
                        _x, dim=1
                ) 
                feats.append(self._extract_feats(_x))
        return th.stack(feats, axis=1)

    def _extract_feats(self, x):
        feats = self._model(x)
        feats = th.flatten(feats, start_dim=1)  # (Bs, -1)
        return feats

transform_img = transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),])
    #transforms.ToTensor()]) # ToTensor() divides by 255

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
        "fps": 10,
        "layer_activation_format": {
            "module.convnet": "TC",
            #"module.convnet.conv1": "CHW",
            #**{f"module.convnet.layer{i}": "CHW" for i in range(1, num_blocks)},
        },
        "duration": None,#(0, 450),
        "time_alignment": "per_frame_aligned",#"evenly_spaced",
        "convert_img_to_video":True,
        "img_duration":450
    }

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."

    wrapper = R3MWrapper(identifier, net, transform_video, 
                                process_output=None,
                                **inferencer_kwargs)
    return wrapper
