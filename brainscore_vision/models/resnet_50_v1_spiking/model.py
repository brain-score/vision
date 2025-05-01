from torchvision.models import resnet50
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from spikingjelly.clock_driven import neuron
import torch
import functools


class ResNet50WithSpikingHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights='IMAGENET1K_V1')
        self.spike = torch.nn.Sequential(
            torch.nn.Identity(),  # Dummy wrapper so we can hook the spike node
            neuron.IFNode()
        )


    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = self.spike(x)  # Now named and hookable
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.resnet, name)


# class ResNet50WithSpikingHead(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.resnet = resnet50(weights='IMAGENET1K_V1')
#         self.spike = neuron.IFNode()  # Insert a spiking node *after* avgpool

#     def forward(self, x):
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)

#         x = self.resnet.layer1(x)
#         x = self.resnet.layer2(x)
#         x = self.resnet.layer3(x)
#         x = self.resnet.layer4(x)

#         x = self.resnet.avgpool(x)  # Brain-Score can still access this
#         x = self.spike(x)  # Spiking head added here (not interfering with exposed layers)
#         x = torch.flatten(x, 1)
#         x = self.resnet.fc(x)
#         return x


def get_model_list():
    return ['resnet_50_v1_spiking']


def get_model(name):
    assert name == 'resnet_50_v1_spiking'
    model = ResNet50WithSpikingHead()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='resnet_50_v1_spiking', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'resnet_50_v1_spiking'
    units = [3, 4, 6, 3]
    layer_names = ['conv1'] + [f'layer{block+1}.{unit}' for block, block_units in
                               enumerate(units) for unit in range(block_units)] + ['avgpool', 'spike']
    return layer_names


def get_bibtex(model_identifier):
    return """
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
@article{spikingjelly,
  title={SpikingJelly: A Reproducible and Extensible Research Framework for Spiking Neural Network},
  author={Fang, Wei et al.},
  journal={arXiv preprint arXiv:2109.13264},
  year={2021}
}
"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)



# from torchvision.models import resnet50
# from brainscore_vision.model_helpers.check_submission import check_models
# from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
# from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
# import functools
# import torch.nn as nn
# from spikingjelly.activation_based import neuron, functional, surrogate
# import torch
# import copy
# import numpy as np

# class SpikingBottleneck(nn.Module):
#     def __init__(self, original_block):
#         super().__init__()
#         # Copy all attributes from original block
#         self.conv1 = original_block.conv1
#         self.bn1 = original_block.bn1
#         self.conv2 = original_block.conv2
#         self.bn2 = original_block.bn2
#         self.conv3 = original_block.conv3
#         self.bn3 = original_block.bn3
#         self.downsample = original_block.downsample
#         self.stride = original_block.stride
        
#         # Replace ReLU with LIF neurons that are more stable for benchmarking
#         # Using a standard surrogate function (API differs between SpikinJelly versions)
#         surrogate_function = surrogate.ATan(alpha=2.0)
        
#         # Create spiking neurons with surrogate gradients
#         self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate_function, step_mode='m')
#         self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate_function, step_mode='m')
#         self.lif3 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate_function, step_mode='m')
        
#         # Membrane scaling to help with numerical stability
#         self.scale_factor = 0.5
        
#     def forward(self, x):
#         # Reset neuron states each time
#         functional.reset_net(self)
        
#         identity = x
        
#         # First block
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.lif1(out * self.scale_factor)
        
#         # Second block
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.lif2(out * self.scale_factor)
        
#         # Third block
#         out = self.conv3(out)
#         out = self.bn3(out)
        
#         # Handle identity connection (downsample if necessary)
#         if self.downsample is not None:
#             identity = self.downsample(x)
        
#         # Add identity and apply final activation
#         out += identity
#         out = self.lif3(out * self.scale_factor)
        
#         # Clip to prevent extreme values that could cause SVD computation issues
#         out = torch.clamp(out, min=-10.0, max=10.0)
        
#         return out

# class ResNetSpikingWrapper(nn.Module):
#     """Wrapper to make the model more stable for Brain-Score benchmarks"""
#     def __init__(self, base_model):
#         super().__init__()
#         # Instead of wrapping, we'll copy the structure to maintain layer access
#         self.conv1 = base_model.conv1
#         self.bn1 = base_model.bn1
#         self.relu = base_model.relu
#         self.maxpool = base_model.maxpool
#         self.layer1 = base_model.layer1
#         self.layer2 = base_model.layer2
#         self.layer3 = base_model.layer3
#         self.layer4 = base_model.layer4
#         self.avgpool = base_model.avgpool
#         self.fc = base_model.fc
        
#     def forward(self, x):
#         # Standard forward pass but with additional safety measures
#         try:
#             # Ensure input values are within a reasonable range
#             x = torch.clamp(x, min=-3.0, max=3.0)
            
#             # Standard ResNet forward pass
#             x = self.conv1(x)
#             x = self.bn1(x)
#             x = self.relu(x)
#             x = self.maxpool(x)

#             x = self.layer1(x)
#             x = self.layer2(x)
#             x = self.layer3(x)
#             x = self.layer4(x)

#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.fc(x)
            
#             # Ensure output is numerically stable
#             if torch.isnan(x).any() or torch.isinf(x).any():
#                 print("Warning: NaN or Inf detected in output, replacing with zeros")
#                 return torch.zeros(x.shape[0], 1000, device=x.device)
                
#             return x
#         except Exception as e:
#             # If an error occurs (like numerical instability), 
#             # return a fallback output that won't break the benchmarks
#             print(f"Warning: Error in forward pass: {e}")
#             return torch.zeros(x.shape[0], 1000, device=x.device)

# def get_model(name):
#     assert name == 'resnet_50_v1_spiking'
    
#     # Load pretrained ResNet-50
#     base_model = resnet50(weights='IMAGENET1K_V1')
    
#     # Replace one block with our spiking implementation
#     # Choosing layer2[1] as it's not too early (would affect feature extraction too much)
#     # and not too late (would have minimal impact)
#     base_model.layer2[1] = SpikingBottleneck(copy.deepcopy(base_model.layer2[1]))
    
#     # Create the model with properly exposed layers
#     model = ResNetSpikingWrapper(base_model)
    
#     # Create preprocessing pipeline
#     preprocessing = functools.partial(load_preprocess_images, image_size=224)
    
#     # Create Brain-Score compatible wrapper
#     wrapper = PytorchWrapper(
#         identifier='resnet_50_v1_spiking', 
#         model=model, 
#         preprocessing=preprocessing,
#     )
#     wrapper.image_size = 224
    
#     return wrapper

# def get_layers(name):
#     assert name == 'resnet_50_v1_spiking'
    
#     # Include all layers that Brain-Score might want to analyze
#     # Structure: 1 conv + [3,4,6,3] blocks + avgpool
#     units = [3, 4, 6, 3]
#     layer_names = ['conv1'] + [
#         f'layer{block+1}.{unit}' 
#         for block, block_units in enumerate(units) 
#         for unit in range(block_units)
#     ] + ['avgpool']
    
#     return layer_names

# def get_bibtex(model_identifier):
#     assert model_identifier == 'resnet_50_v1_spiking'
#     return """
# @inproceedings{he2016deep,
#   title={Deep residual learning for image recognition},
#   author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
#   booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
#   pages={770--778},
#   year={2016}
# }
# @article{fang2021deep,
#   title={Deep learning for spiking neural networks: Algorithm, theory, and implementation},
#   author={Fang, Wentao and Chen, Yanqi and Ding, Jianhao and Chen, Ding and Yu, Zhaofei and Zhou, Huihui and Tian, Yonghong and other},
#   journal={IEEE Transactions on Neural Networks and Learning Systems},
#   year={2021}
# }
# """

# if __name__ == '__main__':
#     check_models.check_base_models(__name__)

# from torchvision.models import resnet50
# from brainscore_vision.model_helpers.check_submission import check_models
# from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
# from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
# import functools
# import torch.nn as nn
# from spikingjelly.activation_based import neuron, functional, surrogate
# import torch
# import copy
# import numpy as np

# class SpikingBottleneck(nn.Module):
#     def __init__(self, original_block):
#         super().__init__()
#         # Copy all attributes from original block
#         self.conv1 = original_block.conv1
#         self.bn1 = original_block.bn1
#         self.conv2 = original_block.conv2
#         self.bn2 = original_block.bn2
#         self.conv3 = original_block.conv3
#         self.bn3 = original_block.bn3
#         self.downsample = original_block.downsample
#         self.stride = original_block.stride
        
#         # Replace ReLU with LIF neurons that are more stable for benchmarking
#         # Using a standard surrogate function (API differs between SpikinJelly versions)
#         surrogate_function = surrogate.ATan(alpha=2.0)
        
#         # Create spiking neurons with surrogate gradients
#         self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate_function, step_mode='m')
#         self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate_function, step_mode='m')
#         self.lif3 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate_function, step_mode='m')
        
#         # Membrane scaling to help with numerical stability
#         self.scale_factor = 0.5
        
#     def forward(self, x):
#         # Reset neuron states each time
#         functional.reset_net(self)
        
#         identity = x
        
#         # First block
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.lif1(out * self.scale_factor)
        
#         # Second block
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.lif2(out * self.scale_factor)
        
#         # Third block
#         out = self.conv3(out)
#         out = self.bn3(out)
        
#         # Handle identity connection (downsample if necessary)
#         if self.downsample is not None:
#             identity = self.downsample(x)
        
#         # Add identity and apply final activation
#         out += identity
#         out = self.lif3(out * self.scale_factor)
        
#         # Clip to prevent extreme values that could cause SVD computation issues
#         out = torch.clamp(out, min=-10.0, max=10.0)
        
#         return out

# class ResNetSpikingWrapper(nn.Module):
#     """Wrapper to make the model more stable for Brain-Score benchmarks"""
#     def __init__(self, base_model):
#         super().__init__()
#         self.base_model = base_model
        
#     def forward(self, x):
#         # Standard forward pass but with additional safety measures
#         try:
#             # Ensure input values are within a reasonable range
#             x = torch.clamp(x, min=-3.0, max=3.0)
#             output = self.base_model(x)
            
#             # Ensure output is numerically stable
#             if torch.isnan(output).any() or torch.isinf(output).any():
#                 print("Warning: NaN or Inf detected in output, replacing with zeros")
#                 return torch.zeros(x.shape[0], 1000, device=x.device)
                
#             return output
#         except Exception as e:
#             # If an error occurs (like numerical instability), 
#             # return a fallback output that won't break the benchmarks
#             print(f"Warning: Error in forward pass: {e}")
#             return torch.zeros(x.shape[0], 1000, device=x.device)

# def get_model(name):
#     assert name == 'resnet_50_v1_spiking'
    
#     # Load pretrained ResNet-50
#     model = resnet50(weights='IMAGENET1K_V1')
    
#     # Replace one block with our spiking implementation
#     # Choosing layer2[1] as it's not too early (would affect feature extraction too much)
#     # and not too late (would have minimal impact)
#     model.layer2[1] = SpikingBottleneck(copy.deepcopy(model.layer2[1]))
    
#     # Wrap the model for stability
#     safe_model = ResNetSpikingWrapper(model)
    
#     # Create preprocessing pipeline
#     preprocessing = functools.partial(load_preprocess_images, image_size=224)
    
#     # Create Brain-Score compatible wrapper
#     wrapper = PytorchWrapper(
#         identifier='resnet_50_v1_spiking', 
#         model=safe_model, 
#         preprocessing=preprocessing,
#     )
#     wrapper.image_size = 224
    
#     return wrapper

# def get_layers(name):
#     assert name == 'resnet_50_v1_spiking'
    
#     # Include all layers that Brain-Score might want to analyze
#     # Structure: 1 conv + [3,4,6,3] blocks + avgpool
#     units = [3, 4, 6, 3]
#     layer_names = ['conv1'] + [
#         f'layer{block+1}.{unit}' 
#         for block, block_units in enumerate(units) 
#         for unit in range(block_units)
#     ] + ['avgpool']
    
#     return layer_names

# def get_bibtex(model_identifier):
#     assert model_identifier == 'resnet_50_v1_spiking'
#     return """
# @inproceedings{he2016deep,
#   title={Deep residual learning for image recognition},
#   author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
#   booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
#   pages={770--778},
#   year={2016}
# }
# @article{fang2021deep,
#   title={Deep learning for spiking neural networks: Algorithm, theory, and implementation},
#   author={Fang, Wentao and Chen, Yanqi and Ding, Jianhao and Chen, Ding and Yu, Zhaofei and Zhou, Huihui and Tian, Yonghong and other},
#   journal={IEEE Transactions on Neural Networks and Learning Systems},
#   year={2021}
# }
# """

# if __name__ == '__main__':
#     check_models.check_base_models(__name__)


# # Safe replacement block for layer2[1]
# class SpikingBottleneck(nn.Module):
#     def __init__(self, original_block):
#         super().__init__()
#         self.conv1 = original_block.conv1
#         self.bn1 = original_block.bn1
#         self.conv2 = original_block.conv2
#         self.bn2 = original_block.bn2
#         self.conv3 = original_block.conv3
#         self.bn3 = original_block.bn3

#         # Spiking neurons with proper step_mode
#         self.relu1 = neuron.LIFNode(step_mode='s')
#         self.relu2 = neuron.LIFNode(step_mode='s')
#         self.relu3 = neuron.LIFNode(step_mode='s')

#     def forward(self, x):
#         # Reset neuron states
#         self.relu1.reset()
#         self.relu2.reset()
#         self.relu3.reset()

#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu2(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         out += identity
#         out = self.relu3(out)

#         return out

# def get_model(name):
# # Modify the model
#     model = resnet50(weights='IMAGENET1K_V1')
#     model.layer2[1] = SpikingBottleneck(model.layer2[1])  # Replace a safe block
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
# @inproceedings{he2016deep,
#   title={Deep residual learning for image recognition},
#   author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
#   booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
#   pages={770--778},
#   year={2016}
# }"""

# if __name__ == '__main__':
#     check_models.check_base_models(__name__)
