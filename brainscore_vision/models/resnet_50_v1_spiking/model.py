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
                # For each ReLU in the block, replace with spiking neuron
                # This maintains the original block structure while adding spiking capabilities
                for name, module in block.named_modules():
                    if isinstance(module, nn.ReLU):
                        # Get the parent module
                        parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                        parent = block if parent_name == '' else getattr(block, parent_name)
                        # Replace the ReLU with a spiking neuron
                        child_name = name if '.' not in name else name.split('.')[-1]
                        setattr(parent, child_name, self.neuron_model(**self.neuron_params))
    
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
        # Set up hooks to capture layer activations
        self.layer_activations = {}
        self.hooks = []
        
        # Initialize the parent class
        super(SpikingResNetWrapper, self).__init__(identifier=identifier, model=model, preprocessing=preprocessing)
        
        # Set up hooks for layers that BrainScore will query
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for all layers needed by BrainScore"""
        def get_hook(name):
            def hook(module, input, output):
                self.layer_activations[name] = output.detach()
            return hook
        
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Register hook for initial conv
        hook = self.model.model.conv1.register_forward_hook(get_hook('conv1'))
        self.hooks.append(hook)
        
        # Register hooks for all blocks in each layer
        for layer_idx, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
            layer = getattr(self.model.model, layer_name)
            
            # Number of blocks in this layer
            num_blocks = len(layer)
            
            for block_idx in range(num_blocks):
                # Get the block at this index
                block = layer[block_idx]
                
                # BrainScore expects layer names in format "layerX.Y"
                brainscore_name = f"{layer_name}.{block_idx}"
                
                # Register the hook on the block
                hook = block.register_forward_hook(get_hook(brainscore_name))
                self.hooks.append(hook)
        
        # Register hook for average pooling
        hook = self.model.model.avgpool.register_forward_hook(get_hook('avgpool'))
        self.hooks.append(hook)
    
    def get_layer(self, layer_name):
        """Get the specified layer module"""
        if layer_name == 'conv1':
            return self.model.model.conv1
        elif layer_name == 'avgpool':
            return self.model.model.avgpool
        elif '.' in layer_name:
            # Handle block layers (layerX.Y format)
            parts = layer_name.split('.')
            if len(parts) == 2:
                layer_name, block_idx = parts
                layer = getattr(self.model.model, layer_name)
                return layer[int(block_idx)]
        
        # If we get here, the layer wasn't found
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def get_activations(self, images, layers):
        """Get activations for specified layers"""
        # Reset activations
        self.layer_activations = {}
        
        # Reset spiking neurons
        functional.reset_net(self.model)
        
        # Forward pass to populate activations
        out_spikes = 0
        for t in range(self.model.T):
            out = self.model(images)
            out_spikes += out
        
        # Collect activations for the requested layers
        result = {}
        for layer_name in layers:
            if layer_name not in self.layer_activations:
                raise KeyError(f"Layer {layer_name} not found in activations. Available layers: {list(self.layer_activations.keys())}")
            result[layer_name] = self.layer_activations[layer_name].cpu().numpy()
        
        return result


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
    
    # Required for BrainScore to find layers
    for layer_name in get_layers(name):
        try:
            wrapper.get_layer(layer_name)
        except Exception as e:
            print(f"Warning: Failed to find layer {layer_name}: {e}")
    
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

# class SpikingBasicBlock(nn.Module):
#     """A spiking version of ResNet's basic block"""
#     def __init__(self, block, neuron_model, **neuron_params):
#         super(SpikingBasicBlock, self).__init__()
#         self.original_block = block
#         self.spiking_layers = []
#         self.neuron_model = neuron_model
#         self.neuron_params = neuron_params
        
#         # Convert appropriate layers to spiking versions
#         self.convert_to_spiking()
        
#     def convert_to_spiking(self):
#         # Keep track of the ReLU layers to replace them with spiking neurons
#         for name, module in self.original_block.named_children():
#             if isinstance(module, nn.ReLU):
#                 setattr(self.original_block, name, self.neuron_model(**self.neuron_params))
#                 self.spiking_layers.append(name)
                
#     def forward(self, x):
#         return self.original_block(x)


# class SpikingResNet(nn.Module):
#     """Converts a standard ResNet to a spiking version"""
#     def __init__(self, model=None, neuron_model=neuron.IFNode, **neuron_params):
#         super(SpikingResNet, self).__init__()
        
#         # Load the original model if not provided
#         if model is None:
#             self.model = resnet50(weights='IMAGENET1K_V1')
#         else:
#             self.model = copy.deepcopy(model)
            
#         # Default neuron parameters if not specified
#         if not neuron_params:
#             neuron_params = {
#                 'v_threshold': 1.0, 
#                 'v_reset': 0.0,
#                 'detach_reset': True,
#                 'surrogate_function': neuron.surrogate.ATan()
#             }
            
#         self.neuron_model = neuron_model
#         self.neuron_params = neuron_params
        
#         # Replace ReLU with spiking neurons in the initial layers
#         self._modify_initial_layers()
        
#         # Convert all residual blocks to spiking versions
#         self._convert_residual_blocks()
        
#         # Add a spiking readout head
#         self._add_spiking_head()
        
#         # Keep track of all hookable layers for brain score
#         self.all_layers = {}
#         self._register_hooks()
        
#         # Settings for temporal dynamics
#         self.T = 4  # Number of time steps for temporal processing
        
#     def _modify_initial_layers(self):
#         # Replace the initial ReLU with a spiking neuron
#         self.model.relu = self.neuron_model(**self.neuron_params)
    
#     def _convert_residual_blocks(self):
#         # Convert all blocks in each layer
#         for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
#             layer_blocks = getattr(self.model, layer_name)
#             for i, block in enumerate(layer_blocks):
#                 # Convert each block to a spiking version
#                 setattr(layer_blocks, str(i), SpikingBasicBlock(block, self.neuron_model, **self.neuron_params))
    
#     def _add_spiking_head(self):
#         # Replace the fully connected layer with a spiking version
#         in_features = self.model.fc.in_features
#         num_classes = self.model.fc.out_features
        
#         # Create a new spiking readout head
#         spiking_head = nn.Sequential(
#             self.neuron_model(**self.neuron_params),
#             nn.Linear(in_features, num_classes)
#         )
        
#         self.model.fc = spiking_head
    
#     def _register_hooks(self):
#         """Register forward hooks to capture activations from specific layers"""
#         def get_activation(name):
#             def hook(model, input, output):
#                 self.all_layers[name] = output
#             return hook
        
#         # Register hooks for all the layers we want to track
#         self.model.conv1.register_forward_hook(get_activation('conv1'))
        
#         # Register hooks for all blocks in each layer
#         for layer_idx, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
#             layer_blocks = getattr(self.model, layer_name)
#             for block_idx in range(len(layer_blocks)):
#                 layer_block = getattr(layer_blocks, str(block_idx))
#                 layer_block.original_block.register_forward_hook(
#                     get_activation(f'{layer_name}.{block_idx}')
#                 )
        
#         # Register hook for average pooling
#         self.model.avgpool.register_forward_hook(get_activation('avgpool'))
    
#     def forward(self, x):
#         # Reset all spiking neurons before forward pass
#         functional.reset_net(self)
        
#         # Run the network for T time steps and accumulate outputs
#         out_spikes = 0
#         for t in range(self.T):
#             out = self.model(x)
#             out_spikes += out
        
#         # Return average response over time steps
#         return out_spikes / self.T


# class SpikingResNetWrapper(PytorchWrapper):
#     """Wrapper for BrainScore compatibility"""
#     def __init__(self, identifier, model, preprocessing):
#         super(SpikingResNetWrapper, self).__init__(identifier=identifier, model=model, preprocessing=preprocessing)
#         self.all_layer_activations = {}
        
#     def get_layer_activations(self, layer_name):
#         """Get activations for a specific layer"""
#         # For blocks, translate from brainscore format to model format
#         if '.' in layer_name:
#             parts = layer_name.split('.')
#             if len(parts) == 2:
#                 block_name, unit_idx = parts
#                 return self.model.all_layers[f'{block_name}.{int(unit_idx)-1}']
        
#         # Direct layer names
#         return self.model.all_layers[layer_name]
    
#     def _get_activations_for_layers(self, images, layers):
#         """Get activations for specified layers"""
#         # Preprocess images
#         device = next(self.model.parameters()).device
#         images = self.preprocess(images).to(device)
        
#         # Forward pass to populate activations
#         with torch.no_grad():
#             self.model(images)
        
#         # Collect activations for requested layers
#         activations = {}
#         for layer in layers:
#             layer_activations = self.get_layer_activations(layer)
#             activations[layer] = layer_activations.detach().cpu().numpy()
            
#         return activations


# # Create the spiking version of ResNet50
# def create_spiking_resnet():
#     original_model = resnet50(weights='IMAGENET1K_V1')
#     spiking_model = SpikingResNet(model=original_model)
#     return spiking_model


# # Functions required by BrainScore

# def get_model(name):
#     assert name == 'resnet_50_v1_spiking'
    
#     # Create the spiking ResNet model
#     model = create_spiking_resnet()
    
#     # Set up preprocessing
#     preprocessing = functools.partial(load_preprocess_images, image_size=224)
    
#     # Create the wrapper
#     wrapper = SpikingResNetWrapper(identifier=name, model=model, preprocessing=preprocessing)
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
# }
#     @article{fang2023incorporating,
#   title={Incorporating learnable membrane time constant to enhance learning of spiking neural networks},
#   author={Fang, Wei and Chen, Yanqi and Ding, Jianhao and Chen, Dongcheng and Yu, Zhaofei and Zhou, Huihui and Tian, Yonghong and other},
#   journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
#   pages={15738--15748},
#   year={2023}
# }
# """


# if __name__ == '__main__':
#     # Use this method to ensure the correctness of the BaseModel implementations.
#     # It executes a mock run of brain-score benchmarks.
#     check_models.check_base_models(__name__)


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