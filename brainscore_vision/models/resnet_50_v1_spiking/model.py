from torchvision.models import resnet50
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import functools
import torch.nn as nn
from spikingjelly.activation_based import neuron
import torch


# ---- Spiking wrapper ----
class SafeSpikingWrapper(nn.Module):
    def __init__(self, block, spike_class=neuron.LIFNode, spike_kwargs=None, clamp_range=(0.0, 1.0)):
        super().__init__()
        self.block = block
        self.spike = spike_class(**(spike_kwargs or {}))
        self.clamp_range = clamp_range

    def forward(self, x):
        out = self.block(x)
        out = torch.clamp(out, min=self.clamp_range[0], max=self.clamp_range[1])
        out = self.spike(out)
        if not torch.isfinite(out).all():
            raise ValueError("SafeSpikingWrapper: model output contains NaNs or Infs.")
        return out

    def reset(self):
        if hasattr(self.spike, 'reset'):
            self.spike.reset()

# ---- Reset all spiking neurons before inference ----
def reset_all_spiking(model):
    for m in model.modules():
        if hasattr(m, 'reset'):
            m.reset()

# ---- Brain-Score model wrapper ----
def get_model(name):
    assert name == 'resnet_50_v1_spiking'
    model = resnet50(weights='IMAGENET1K_V1')

    # Wrap selected layers with spiking
    spike_class = neuron.LIFNode
    spike_params = {'tau': 2.0, 'v_threshold': 1.0, 'v_reset': 0.0}

    # Choose safe blocks without downsample
    model.layer2[1] = SafeSpikingWrapper(model.layer2[1], spike_class, spike_params)
    model.layer2[2] = SafeSpikingWrapper(model.layer2[2], spike_class, spike_params)

    # Wrap the model in a class that resets spiking neurons before inference
    class ResettingModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            reset_all_spiking(self.model)
            return self.model(x)

    wrapped_model = ResettingModelWrapper(model)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)

    wrapper = PytorchWrapper(identifier='resnet_50_v1_spiking', model=wrapped_model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'resnet_50_v1_spiking'
    units = [3, 4, 6, 3]
    layer_names = ['conv1'] + [f'layer{block+1}.{unit}' for block, block_units in
                               enumerate(units) for unit in range(block_units)] + ['avgpool']
    return layer_names


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
