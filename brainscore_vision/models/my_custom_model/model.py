from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import torchvision
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from PIL import Image

# This is an example implementation for submitting custom model named my_custom_model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.

"""
class MyCustomModel(torch.nn.Module):
    def __init__(self):
        super(MyCustomModel, self).__init__()
        model = torchvision.models.resnet18(weights=None, num_classes=100) # define model
        checkpoint = torch.load("/Users/dunhan/Desktop/topoV4/ShapePcl/checkpoints/checkpoint_ShapePcl.pth.tar", map_location=torch.device('cpu'))["state_dict"] # load checkpoint
        model.load_state_dict(checkpoint) # load checkpoint
        self.resnet18 = model

    def forward(self, x):
        return self.resnet18(x)
"""

def get_model_list():
    return ['resnet18_moco']


def transforms():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

def custom_image_preprocess(images, transforms):
    # print(images, transforms)
    images = [Image.open(image).convert("RGB") for image in images]
    images = [transforms(image) for image in images]
    images = [np.array(image) for image in images]
    images = np.stack(images)
    return images

def get_model(name):
    assert name == 'resnet18_moco'
    preprocessing = functools.partial(
        custom_image_preprocess, transforms=transforms()
    )
    model = torchvision.models.resnet18(weights=None, num_classes=100) # define model
    checkpoint_path = "./models/my_custom_model/moco_IN100_400.pth.tar" # resnet18_moco
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"] # load checkpoint
    model.load_state_dict(checkpoint) # load checkpoint
    model.eval()
    activations_model = PytorchWrapper(identifier='resnet18_moco', model=model, preprocessing=preprocessing)
    activations_model.image_size = 224
    model = ModelCommitment(identifier='resnet18_moco', 
                            activations_model=activations_model, 
                            layers=["layer1.0", "layer1.1", "layer2.0", "layer2.1", "layer3.0", "layer3.1", "layer4.0", "layer4.1"])
    return activations_model


def get_layers(name):
    assert name == 'resnet18_moco'
    return ["layer1.0", "layer1.1", "layer2.0", "layer2.1", "layer3.0", "layer3.1", "layer4.0", "layer4.1"]


def get_bibtex(model_identifier):
    return """resnet18_moco"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)

