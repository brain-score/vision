from torchvision.models import resnet50
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import functools
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional

# Optional: define a spiking head (not scored unless added to get_layers)
class SpikingHead(nn.Module):
    def __init__(self, input_dim=2048, output_dim=1000, T=4):
        super().__init__()
        self.T = T
        self.fc = nn.Linear(input_dim, output_dim)
        self.spike_layer = neuron.LIFNode()

    def forward(self, x):
        # x shape: [batch, input_dim]
        x = x.unsqueeze(0).repeat(self.T, 1, 1)  # [T, batch, input_dim]
        out_spike = 0
        for t in range(self.T):
            out = self.fc(x[t])
            out = self.spike_layer(out)
            out_spike += out
        functional.reset_net(self)
        return out_spike / self.T


# Replace ReLU with spiking neurons in selected parts of the model
def replace_relu_with_spiking(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, neuron.LIFNode(step_mode='s', backend='torch'))
        else:
            replace_relu_with_spiking(child)



# Brain-Score interface
def get_model(name):
        # Load model and modify
    model = resnet50(weights='IMAGENET1K_V1')
    replace_relu_with_spiking(model.layer3)
    replace_relu_with_spiking(model.layer4)
    model.fc = SpikingHead(input_dim=2048, output_dim=1000)  # not scored, but included for completeness
    assert name == 'resnet_50_v1_spiking'
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