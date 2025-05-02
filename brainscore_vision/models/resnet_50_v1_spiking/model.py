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
        # Load pretrained ResNet50
        resnet = resnet50(weights='IMAGENET1K_V1')
        
        # Extract the layers from the pretrained model
        # and register them directly in this model
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet blocks
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Final layers
        self.avgpool = resnet.avgpool
        self.spike = neuron.IFNode()  # Spiking neuron layer
        self.fc = resnet.fc  # Final fully connected layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.spike(x)  # Apply spiking neuron
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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


