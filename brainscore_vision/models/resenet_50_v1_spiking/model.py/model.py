import torch
import torch.nn as nn
import functools
from torchvision.models import resnet50
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import spikingjelly.clock_driven.neuron as neuron
import spikingjelly.clock_driven.layer as layer
import spikingjelly.clock_driven.functional as functional
import copy


class SpikingBasicBlock(nn.Module):
    """A spiking version of ResNet's basic block"""
    def __init__(self, block, neuron_model, **neuron_params):
        super(SpikingBasicBlock, self).__init__()
        self.original_block = block
        self.spiking_layers = []
        self.neuron_model = neuron_model
        self.neuron_params = neuron_params
        
        # Convert appropriate layers to spiking versions
        self.convert_to_spiking()
        
    def convert_to_spiking(self):
        # Keep track of the ReLU layers to replace them with spiking neurons
        for name, module in self.original_block.named_children():
            if isinstance(module, nn.ReLU):
                setattr(self.original_block, name, self.neuron_model(**self.neuron_params))
                self.spiking_layers.append(name)
                
    def forward(self, x):
        return self.original_block(x)


class SpikingResNet(nn.Module):
    """Converts a standard ResNet to a spiking version"""
    def __init__(self, model=None, neuron_model=neuron.IFNode, **neuron_params):
        super(SpikingResNet, self).__init__()
        
        # Load the original model if not provided
        if model is None:
            self.model = resnet50(weights='IMAGENET1K_V1')
        else:
            self.model = copy.deepcopy(model)
            
        # Default neuron parameters if not specified
        if not neuron_params:
            neuron_params = {
                'v_threshold': 1.0, 
                'v_reset': 0.0,
                'detach_reset': True,
                'surrogate_function': neuron.surrogate.ATan()
            }
            
        self.neuron_model = neuron_model
        self.neuron_params = neuron_params
        
        # Replace ReLU with spiking neurons in the initial layers
        self._modify_initial_layers()
        
        # Convert all residual blocks to spiking versions
        self._convert_residual_blocks()
        
        # Add a spiking readout head
        self._add_spiking_head()
        
        # Keep track of all hookable layers for brain score
        self.all_layers = {}
        self._register_hooks()
        
        # Settings for temporal dynamics
        self.T = 4  # Number of time steps for temporal processing
        
    def _modify_initial_layers(self):
        # Replace the initial ReLU with a spiking neuron
        self.model.relu = self.neuron_model(**self.neuron_params)
    
    def _convert_residual_blocks(self):
        # Convert all blocks in each layer
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer_blocks = getattr(self.model, layer_name)
            for i, block in enumerate(layer_blocks):
                # Convert each block to a spiking version
                setattr(layer_blocks, str(i), SpikingBasicBlock(block, self.neuron_model, **self.neuron_params))
    
    def _add_spiking_head(self):
        # Replace the fully connected layer with a spiking version
        in_features = self.model.fc.in_features
        num_classes = self.model.fc.out_features
        
        # Create a new spiking readout head
        spiking_head = nn.Sequential(
            self.neuron_model(**self.neuron_params),
            nn.Linear(in_features, num_classes)
        )
        
        self.model.fc = spiking_head
    
    def _register_hooks(self):
        """Register forward hooks to capture activations from specific layers"""
        def get_activation(name):
            def hook(model, input, output):
                self.all_layers[name] = output
            return hook
        
        # Register hooks for all the layers we want to track
        self.model.conv1.register_forward_hook(get_activation('conv1'))
        
        # Register hooks for all blocks in each layer
        for layer_idx, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
            layer_blocks = getattr(self.model, layer_name)
            for block_idx in range(len(layer_blocks)):
                layer_block = getattr(layer_blocks, str(block_idx))
                layer_block.original_block.register_forward_hook(
                    get_activation(f'{layer_name}.{block_idx}')
                )
        
        # Register hook for average pooling
        self.model.avgpool.register_forward_hook(get_activation('avgpool'))
    
    def forward(self, x):
        # Reset all spiking neurons before forward pass
        functional.reset_net(self)
        
        # Run the network for T time steps and accumulate outputs
        out_spikes = 0
        for t in range(self.T):
            out = self.model(x)
            out_spikes += out
        
        # Return average response over time steps
        return out_spikes / self.T


class SpikingResNetWrapper(PytorchWrapper):
    """Wrapper for BrainScore compatibility"""
    def __init__(self, identifier, model, preprocessing):
        super(SpikingResNetWrapper, self).__init__(identifier=identifier, model=model, preprocessing=preprocessing)
        self.all_layer_activations = {}
        
    def get_layer_activations(self, layer_name):
        """Get activations for a specific layer"""
        # For blocks, translate from brainscore format to model format
        if '.' in layer_name:
            parts = layer_name.split('.')
            if len(parts) == 2:
                block_name, unit_idx = parts
                return self.model.all_layers[f'{block_name}.{int(unit_idx)-1}']
        
        # Direct layer names
        return self.model.all_layers[layer_name]
    
    def _get_activations_for_layers(self, images, layers):
        """Get activations for specified layers"""
        # Preprocess images
        device = next(self.model.parameters()).device
        images = self.preprocess(images).to(device)
        
        # Forward pass to populate activations
        with torch.no_grad():
            self.model(images)
        
        # Collect activations for requested layers
        activations = {}
        for layer in layers:
            layer_activations = self.get_layer_activations(layer)
            activations[layer] = layer_activations.detach().cpu().numpy()
            
        return activations


# Create the spiking version of ResNet50
def create_spiking_resnet():
    original_model = resnet50(weights='IMAGENET1K_V1')
    spiking_model = SpikingResNet(model=original_model)
    return spiking_model


# Functions required by BrainScore

def get_model(name):
    assert name == 'resnet_50_v1_spiking'
    
    # Create the spiking ResNet model
    model = create_spiking_resnet()
    
    # Set up preprocessing
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    
    # Create the wrapper
    wrapper = SpikingResNetWrapper(identifier='resnet_50_v1_spiking', model=model, preprocessing=preprocessing)
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
    @article{fang2023incorporating,
  title={Incorporating learnable membrane time constant to enhance learning of spiking neural networks},
  author={Fang, Wei and Chen, Yanqi and Ding, Jianhao and Chen, Dongcheng and Yu, Zhaofei and Zhou, Huihui and Tian, Yonghong and other},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15738--15748},
  year={2023}
}
"""


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
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