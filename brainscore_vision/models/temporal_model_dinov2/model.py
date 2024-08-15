from r3m_model import pfDINOV2
import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from torchvision import transforms


class DINOV2Wrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.permute(0, 2, 1, 3, 4)
        tensor = tensor.to(self._device)
        return self._model(tensor)  # encoder only

transform_img = transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),])
    #transforms.ToTensor()]) # ToTensor() divides by 255

def transform_video(video):
    import torch
    frames = torch.Tensor(video.to_numpy()).permute(0, 3, 1, 2)
    frames = transform_img(frames)
    return frames.permute(1, 0, 2, 3)


def get_model(identifier, num_frames=16):
    assert identifier.startswith("DINO")
    
    if identifier == "DINO-GIANT-Temporal":
        model_name = "facebook/dinov2-giant"
    if identifier == "DINO-LARGE-Temporal":
        model_name = "facebook/dinov2-large"
    elif identifier == "DINO-BASE-Temporal":
        model_name = "facebook/dinov2-base"

    # Instantiate the model

    net = pfDINOV2(model_name)

    inferencer_kwargs = {
        "fps": 10,
        "layer_activation_format": {
            "encoder": "TC",
        },
        "duration": None,#(0, 450),
        "time_alignment": "per_frame_aligned",#"evenly_spaced",
        "convert_img_to_video":True,
        "img_duration":450
    }

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."

    def process_activation(layer, layer_name, inputs, output):
        output = th.stack(output, axis=1)
        return output
    
    wrapper = DINOV2Wrapper(identifier, net, transform_video, 
                                process_output=process_activation,
                                **inferencer_kwargs)
    return wrapper
