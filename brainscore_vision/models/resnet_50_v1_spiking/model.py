from torchvision.models import resnet50
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import functools
import torch
import torch.nn as nn

# Simple custom spiking neuron implementation
class SimpleLIF(nn.Module):
    """
    A simplified Leaky Integrate-and-Fire neuron with automatic tensor shape handling.
    This implementation is designed to be more robust to tensor shape changes.
    """
    def __init__(self, threshold=1.0, reset_value=0.0, tau=2.0, detach_reset=True):
        super().__init__()
        self.threshold = threshold
        self.reset_value = reset_value
        self.tau = tau
        self.detach_reset = detach_reset
        self.v = None  # Membrane potential

    def forward(self, x):
        # Create a new membrane potential tensor if None or shape mismatch
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        
        # Update membrane potential
        self.v = self.v + (x - (self.v - self.reset_value)) / self.tau
        
        # Generate spikes
        spike = (self.v >= self.threshold).float()
        
        # Reset membrane potential
        if self.detach_reset:
            self.v = (1.0 - spike) * self.v + spike * self.reset_value
        else:
            self.v = self.v * (1.0 - spike) + spike * self.reset_value
            
        return spike

    def reset(self):
        self.v = None  # Reset membrane potential

class SimplifiedSpikingBlock(nn.Module):
    """
    A simple block that adds spiking functionality to the model.
    This can be inserted at various points in the architecture.
    """
    def __init__(self, threshold=1.0, tau=2.0):
        super().__init__()
        self.lif = SimpleLIF(threshold=threshold, tau=tau)
        
    def forward(self, x):
        return self.lif(x)

def replace_relu_with_spiking(module):
    """Replace ReLU with custom spiking neurons"""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, SimpleLIF())
        else:
            replace_relu_with_spiking(child)

def convert_to_spiking_model(model, spiking_layers=None):
    """
    Converts specified layers of a model to use spiking neurons in-place.
    Returns the modified model (same instance).
    """
    # Default to converting specific layers if not specified
    if spiking_layers is None:
        spiking_layers = ['layer3', 'layer4']
    
    # Replace ReLUs with spiking neurons in specified layers
    for layer_name in spiking_layers:
        if hasattr(model, layer_name):
            replace_relu_with_spiking(getattr(model, layer_name))
    
    # Create a forward hook to reset neurons before each forward pass
    def reset_neurons_hook(module, input):
        for m in module.modules():
            if isinstance(m, SimpleLIF):
                m.reset()
        return None
    
    # Register the hook
    model.register_forward_pre_hook(reset_neurons_hook)
    return model

# Add a new spiking layer between fc and the final output
def add_spiking_output_layer(model):
    """Add a spiking layer after the final fully connected layer"""
    original_fc = model.fc
    new_fc = nn.Sequential(
        original_fc,
        SimplifiedSpikingBlock()
    )
    model.fc = new_fc
    return model

def get_model(name):
    assert name == 'resnet_50_v1_spiking'

    # Load base model
    model = resnet50(weights='IMAGENET1K_V1')
    
    # Option 1: Convert existing layers to spiking neurons
    convert_to_spiking_model(model, spiking_layers=['layer3', 'layer4'])
    
    # Option 2 (commented out): Add a spiking layer to the output
    # add_spiking_output_layer(model)
    
    # Create preprocessing and wrapper
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='resnet_50_v1_spiking', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'resnet_50_v1_spiking'
    units = [3, 4, 6, 3]
    
    # Standard ResNet layers
    layer_names = ['conv1'] + [f'layer{block+1}.{unit}' for block, block_units in
                              enumerate(units) for unit in range(block_units)] + ['avgpool']
    
    # Uncomment below to add the new spiking output layer for scoring
    # layer_names.append('fc.1')  # This would reference the added spiking layer after fc
    
    return layer_names

def get_bibtex(model_identifier):
    assert model_identifier == 'resnet_50_v1_spiking'
    return """
    @inproceedings{he2016deep,
      title={Deep residual learning for image recognition},
      author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
      booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages={770--778},
      year={2016}
    }"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)



# from torchvision.models import resnet50
# from brainscore_vision.model_helpers.check_submission import check_models
# from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
# from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
# from brainscore_vision.model_helpers.check_submission import check_models
# import functools


# model = resnet50(weights='IMAGENET1K_V1')

# def get_model(name):
#     assert name == 'resnet_50_v1_spiking'
#     preprocessing = functools.partial(load_preprocess_images, image_size=224)
#     wrapper = PytorchWrapper(identifier='resnet_50_v1_spiking', model=model, preprocessing=preprocessing)
#     wrapper.image_size = 224
#     return wrapper

# def get_layers(name):
#     assert name == 'resnet_50_v1_spiking'
#     units = [3, 4, 6, 3]
#     layer_names = ['conv1'] + [f'layer{block+1}.{unit}' for block, block_units in
#                                enumerate(units) for unit in range(block_units)] + ['avgpool']
#     return layer_names


# def get_bibtex(model_identifier):
#     assert model_identifier == 'resnet_50_v1_spiking'
#     return """
#     @inproceedings{he2016deep,
#   title={Deep residual learning for image recognition},
#   author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
#   booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
#   pages={770--778},
#   year={2016}
# }"""



# if __name__ == '__main__':
#     # Use this method to ensure the correctness of the BaseModel implementations.
#     # It executes a mock run of brain-score benchmarks.
#     check_models.check_base_models(__name__)