from torchvision.models import resnet50
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import functools
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
import copy

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
            if isinstance(m, neuron.BaseNode):
                m.reset()
        return None
    
    # Register the hook
    model.register_forward_pre_hook(reset_neurons_hook)
    return model

def replace_relu_with_spiking(module):
    """Replace ReLU with properly configured LIF neurons"""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            # Configure LIF neurons with appropriate parameters
            # Using surrogate gradient for better training stability
            lif = neuron.LIFNode(
                tau=2.0,  # Time constant
                v_threshold=1.0,  # Firing threshold
                v_reset=0.0,  # Reset potential
                surrogate_function=surrogate.ATan(),  # Surrogate gradient function
                detach_reset=True,  # Detach reset for training stability
                step_mode='s',  # Single-step mode
                backend='torch'  # Pure PyTorch backend for compatibility
            )
            setattr(module, name, lif)
        else:
            replace_relu_with_spiking(child)

def get_model(name):
    assert name == 'resnet_50_v1_spiking'

    # Load base model
    model = resnet50(weights='IMAGENET1K_V1')
    
    # Convert to spiking in-place (only layer3 and layer4)
    convert_to_spiking_model(model, spiking_layers=['layer3', 'layer4'])
    
    # Create preprocessing and wrapper
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='resnet_50_v1_spiking', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'resnet_50_v1_spiking'
    units = [3, 4, 6, 3]
    layer_names = ['conv1'] + [f'layer{block+1}.{unit}' for block, block_units in
                              enumerate(units) for unit in range(block_units)] + ['avgpool']
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
    }
    @article{fang2020incorporating,
      title={Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks},
      author={Fang, Wei and Yu, Zhaofei and Chen, Yanqi and Masquelier, Timothée and Huang, Tiejun and Tian, Yonghong},
      journal={arXiv preprint arXiv:2007.05785},
      year={2020}
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