from r3m_model import pfResNet
import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from torchvision import transforms


class ResNetrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.permute(0, 2, 1, 3, 4)
        tensor = tensor.to(self._device)
        return self._model(tensor)  # encoder only

# Define the ImageNet mean and std
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Define the transform
transform_img = transforms.Compose([
    # Resize the image to the size expected by ViT-MAE
    transforms.Resize((224, 224)),  # Example size for ViT

    # Convert the image to a tensor
    transforms.ToTensor(),

    # Normalize the image with ImageNet mean and std
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    # Optional: Add more augmentations if needed
    # For example, RandomHorizontalFlip, ColorJitter, etc.
])

def transform_video(video):
    import torch
    frames = torch.Tensor(video.to_numpy()).permute(0, 3, 1, 2)
    frames = transform_img(frames)
    return frames.permute(1, 0, 2, 3)


def get_model(identifier, num_frames=16):
    assert identifier.startswith("R3M")

    if identifier == "ResNet50-Temporal":
        model_name = "microsoft/resnet-50"
    elif identifier == "ResNet18-Temporal":
        model_name = "microsoft/resnet-18"
    elif identifier == "ResNet34-Temporal":
        model_name = "microsoft/resnet-34"
    elif identifier == "ResNet101-Temporal":
        model_name = "microsoft/resnet-101"
    elif identifier == "ResNet152-Temporal":
        model_name = "microsoft/resnet-152"

    # Instantiate the model

    net = pfResNet(model_name)

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
    
    wrapper = ResNetrapper(identifier, net, transform_video, 
                                process_output=process_activation,
                                **inferencer_kwargs)
    return wrapper