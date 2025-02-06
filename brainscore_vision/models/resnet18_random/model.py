import torch
from torchvision.models import resnet18
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import functools

# Define preprocessing (resize to 224x224 as required by ResNet)
preprocessing = functools.partial(load_preprocess_images, image_size=224)

# Define ResNet18 with random weights
def get_model(name):
    assert name == 'resnet18_random'
    # Load ResNet18 without pre-trained weights
    model = resnet18(pretrained=False)
    # Wrap the model with Brain-Score's PytorchWrapper
    activations_model = PytorchWrapper(identifier='resnet18_random', model=model, preprocessing=preprocessing)
    return ModelCommitment(
        identifier='resnet18_random',
        activations_model=activations_model,
        # Specify layers for evaluation
        layers=['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    )

# Specify layers to test
def get_layers(name):
    assert name == 'resnet18_random'
    return ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

# Optional: Provide a BibTeX reference for the model
def get_bibtex(model_identifier):
    return """
    @misc{resnet18_test_consistency,
    title={ResNet18 with Random Weights},
    author={Clear Glue},
    year={2024},
    }
    """

if __name__ == '__main__':
    from brainscore_vision.model_helpers.check_submission import check_models
    check_models.check_base_models(__name__)
