from torchvision.models import resnet50
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from spikingjelly.clock_driven import neuron
import torch
import functools

class ResNet50WithSpikingHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights='IMAGENET1K_V1')
        self.spike = neuron.IFNode()  # Insert a spiking node *after* avgpool

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)  # Brain-Score can still access this
        x = self.spike(x)  # Spiking head added here (not interfering with exposed layers)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


def get_model_list():
    return ['resnet_50_spiking']


def get_model(name):
    assert name == 'resnet_50_spiking'
    model = ResNet50WithSpikingHead()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='resnet_50_spiking', model=model, preprocessing=preprocessing)# reuse the full set of original layers
    wrapper.image_size = 224
    return model


def get_layers(name):
    assert name == 'resnet_50_spiking'
    units = [3, 4, 6, 3]
    layer_names = ['conv1'] + [f'layer{block+1}.{unit}' for block, block_units in
                               enumerate(units) for unit in range(block_units)] + ['avgpool']
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
# from brainscore_vision.model_helpers.check_submission import check_models
# import functools


# model = resnet50(weights='IMAGENET1K_V1')

# def get_model(name):
#     assert name == 'resnet_50_v1'
#     preprocessing = functools.partial(load_preprocess_images, image_size=224)
#     wrapper = PytorchWrapper(identifier='resnet_50_v1', model=model, preprocessing=preprocessing)
#     wrapper.image_size = 224
#     return wrapper

# def get_layers(name):
#     assert name == 'resnet_50_v1'
#     units = [3, 4, 6, 3]
#     layer_names = ['conv1'] + [f'layer{block+1}.{unit}' for block, block_units in
#                                enumerate(units) for unit in range(block_units)] + ['avgpool']
#     return layer_names


# def get_bibtex(model_identifier):
#     assert model_identifier == 'resnet_50_v1'
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