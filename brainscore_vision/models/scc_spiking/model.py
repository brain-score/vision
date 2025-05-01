# from brainscore_vision.model_helpers.check_submission import check_models
# import functools
# from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
# from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
# import torch
# import numpy as np
# from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
# from spikingjelly.activation_based import neuron, surrogate, functional


# class RobustSpikingModel(torch.nn.Module):
#     def __init__(self, time_steps=10):
#         super(RobustSpikingModel, self).__init__()
#         # Number of time steps for temporal processing
#         self.time_steps = time_steps
        
#         # Use surrogate gradient function for better training stability
#         spike_fn = surrogate.ATan()
        
#         # Define layers with more channels for better representational capacity
#         self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
#         self.bn1 = torch.nn.BatchNorm2d(16)  # Stabilize activations
#         self.spike1 = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0, 
#                                      surrogate_function=spike_fn, step_mode='m')

#         # Add a second convolutional layer
#         self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
#         self.bn2 = torch.nn.BatchNorm2d(32)
#         self.spike2 = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
#                                      surrogate_function=spike_fn, step_mode='m')
        
#         # Add pooling to reduce dimensionality
#         self.pool = torch.nn.AvgPool2d(kernel_size=2)
        
#         # Calculate the size after convolutions and pooling
#         # 224 -> conv1(3x3) -> 222 -> conv2(3x3) -> 220 -> pool(2x2) -> 110
#         feature_map_size = 110
#         linear_input_size = feature_map_size * feature_map_size * 32
        
#         # Add a hidden layer before the final output
#         self.fc1 = torch.nn.Linear(linear_input_size, 2048)
#         self.bn3 = torch.nn.BatchNorm1d(2048)
#         self.spike3 = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
#                                      surrogate_function=spike_fn, step_mode='m')
        
#         # Output layer for ImageNet-like classification
#         self.fc2 = torch.nn.Linear(2048, 1000)
#         self.spike_out = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
#                                        surrogate_function=spike_fn, step_mode='m')
        
#         # Initialize weights for better numerical stability
#         self._initialize_weights()
        
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, torch.nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     torch.nn.init.constant_(m.bias, 0)
#             elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
#                 torch.nn.init.constant_(m.weight, 1)
#                 torch.nn.init.constant_(m.bias, 0)
#             elif isinstance(m, torch.nn.Linear):
#                 torch.nn.init.normal_(m.weight, 0, 0.01)
#                 torch.nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # Reset all spiking neurons
#         functional.reset_net(self)
        
#         # Reshape input for multi-step processing if needed
#         batch_size = x.size(0)
#         if len(x.shape) == 4:  # Standard input (batch, channels, height, width)
#             # Repeat the input across time dimension for spiking processing
#             x = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)  # [T, B, C, H, W]
        
#         # Process through the network with temporal dynamics
#         outputs = []
#         for t in range(self.time_steps):
#             x_t = x[t] if len(x.shape) == 5 else x
            
#             # First convolutional block
#             x_t = self.conv1(x_t)
#             x_t = self.bn1(x_t)
#             x_t = self.spike1(x_t)
            
#             # Second convolutional block
#             x_t = self.conv2(x_t)
#             x_t = self.bn2(x_t)
#             x_t = self.spike2(x_t)
#             x_t = self.pool(x_t)
            
#             # Flatten features for fully connected layers
#             x_t = x_t.view(batch_size, -1)
            
#             # Fully connected layers
#             x_t = self.fc1(x_t)
#             x_t = self.bn3(x_t)
#             x_t = self.spike3(x_t)
            
#             x_t = self.fc2(x_t)
#             x_t = self.spike_out(x_t)
            
#             # Ensure no NaN values are present
#             x_t = torch.where(torch.isnan(x_t), torch.zeros_like(x_t), x_t)
            
#             # Add small epsilon for numerical stability
#             x_t = x_t + 1e-8
            
#             outputs.append(x_t)
        
#         # Average across time steps for final output
#         result = torch.stack(outputs).mean(dim=0)
        
#         # Apply normalization to improve numerical stability
#         result = torch.nn.functional.normalize(result, p=2, dim=1)
        
#         return result
    
#     # Add explicit layer hooks for BrainScore
#     def get_all_layers(self):
#         return {
#             'conv1': self.conv1,
#             'bn1': self.bn1,
#             'spike1': self.spike1,
#             'conv2': self.conv2,
#             'bn2': self.bn2,
#             'spike2': self.spike2, 
#             'pool': self.pool,
#             'fc1': self.fc1,
#             'bn3': self.bn3,
#             'spike3': self.spike3,
#             'fc2': self.fc2,
#             'spike_out': self.spike_out
#         }


# # Functions required by BrainScore
# def get_model_list():
#     return ['simple_spiking_model']


# def get_model(name):
#     assert name == 'simple_spiking_model'
#     preprocessing = functools.partial(load_preprocess_images, image_size=224)
#     model = RobustSpikingModel(time_steps=10)
#     wrapper = PytorchWrapper(identifier='simple_spiking_model', model=model, preprocessing=preprocessing)
#     wrapper.image_size = 224
#     return wrapper


# def get_layers(name):
#     assert name == 'simple_spiking_model'
#     # Focus on visual processing layers that are most relevant for brain benchmarks
#     # Exclude purely computational layers like batch normalization
#     # Include convolutional layers, spiking activations, and pooling operations
#     return ['conv1', 'spike1', 'conv2', 'spike2', 'pool']


# def get_bibtex(model_identifier):
#     return """
#     @article{spikingjelly,
#         title={SpikingJelly: An open-source deep learning framework for spiking neural networks},
#         author={Fang, Wei and Chen, Yanqi and Ding, Jianhao and Chen, Duzhen and Yu, Zhaofei and Zhou, Huihui and Tian, Yonghong and other contributors},
#         journal={arXiv preprint arXiv:2202.12729},
#         year={2022}
#     }
#     """


# if __name__ == '__main__':
#     check_models.check_base_models(__name__)



from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

# Import from SpikingJelly
from spikingjelly.clock_driven import neuron, encoding, functional, layer

class MySpikingModel(torch.nn.Module):
    def __init__(self):
        super(MySpikingModel, self).__init__()
        # Example: Poisson encoding and a simple spiking layer
        self.encoder = encoding.PoissonEncoder()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=3)
        self.sn = neuron.IFNode()  # Integrate-and-fire neuron
        self.fc = torch.nn.Linear((224 - 2)**2 * 2, 1000)

    def forward(self, x):
        x = x / 255.0  # normalize if needed
        # Repeat input across time steps (T, N, C, H, W)
        T = 10  # number of time steps
        x = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        x = self.encoder(x)
        spk_out = []

        for t in range(T):
            out = self.conv(x[t])
            out = self.sn(out)
            spk_out.append(out)

        out = sum(spk_out) / T  # average across time
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model_list():
    return ['simple_spiking_model']

def get_model(name):
    assert name == 'simple_spiking_model'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    model_instance = MySpikingModel()
    wrapper = PytorchWrapper(identifier='simple_spiking_model', model=model_instance, preprocessing=preprocessing)
    model = ModelCommitment(identifier='simple_spiking_model', activations_model=wrapper,
                            layers=['conv', 'sn'])  # list only relevant layers
    wrapper.image_size = 224
    return model

def get_layers(name):
    assert name == 'simple_spiking_model'
    return ['conv', 'sn']  # only layers Brain-Score will evaluate

def get_bibtex(model_identifier):
    return """@article{spikingjelly,
    title={SpikingJelly: A Reproducible and Extensible Research Framework for Spiking Neural Network},
    author={Fang, Wei et al.},
    journal={arXiv preprint arXiv:2109.13264},
    year={2021}
    }"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)










#### SIMPLE SPIKING MODEL
# from brainscore_vision.model_helpers.check_submission import check_models
# import functools
# from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
# from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
# import torch
# import numpy as np
# from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
# from spikingjelly.activation_based import neuron


# class SimpleSpikingModel(torch.nn.Module):
#     def __init__(self):
#         super(SimpleSpikingModel, self).__init__()
#         self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
#         self.spike1 = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0, step_mode='s')
#         conv_output_size = (224 - 3 + 1)  # 222
#         linear_input_size = conv_output_size * conv_output_size * 2
#         self.linear = torch.nn.Linear(linear_input_size, 1000)
#         self.spike2 = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0, step_mode='s')

#     def forward(self, x):
#         # Reset membrane potentials at the start of every forward pass
#         for module in self.modules():
#             if hasattr(module, 'reset'):
#                 module.reset()

#         x = self.conv1(x)
#         x = self.spike1(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear(x)
#         x = self.spike2(x)
#         return x


# def get_model_list():
#     return ['simple_spiking_model']


# def get_model(name):
#     assert name == 'simple_spiking_model'
#     preprocessing = functools.partial(load_preprocess_images, image_size=224)
#     model = SimpleSpikingModel()
#     wrapper = PytorchWrapper(identifier='simple_spiking_model', model=model, preprocessing=preprocessing)
#     wrapper.image_size = 224
#     return wrapper


# def get_layers(name):
#     assert name == 'simple_spiking_model'
#     return ['conv1', 'spike1', 'spike2']


# def get_bibtex(model_identifier):
#     return """
#     """


# if __name__ == '__main__':
#     check_models.check_base_models(__name__)
