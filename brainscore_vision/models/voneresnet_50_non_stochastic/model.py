import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import torch
from torch.nn import Module
import torch.nn as nn
from brainscore_vision.model_helpers.s3 import load_weight_file
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import ssl

# needed import for globals() to work!
from .vonenet.vonenet.vonenet import VOneNet

ssl._create_default_https_context = ssl._create_unverified_context


class Wrapper(Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model


def get_model_from_s3():
    model_arch = 'resnet50'
    pretrained = True
    if pretrained and model_arch:
        weights_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                        relative_path="voneresnet-50-non_stochastic/voneresnet50_ns_e70.pth.tar",
                                        version_id="Vgdhpj8hBvUvpV.3bTHyJCUaoG93hxF.",
                                        sha1="c270528818d6d7fc67a6aec86919d47311ad6221")
        ckpt_data = torch.load(weights_path, map_location=torch.device('cpu'))
        stride = ckpt_data['flags']['stride']
        simple_channels = ckpt_data['flags']['simple_channels']
        complex_channels = ckpt_data['flags']['complex_channels']
        k_exc = ckpt_data['flags']['k_exc']

        noise_mode = ckpt_data['flags']['noise_mode']
        noise_scale = ckpt_data['flags']['noise_scale']
        noise_level = ckpt_data['flags']['noise_level']

        model_id = ckpt_data['flags']['arch'].replace('_', '').lower()

        model = globals()[f'VOneNet'](model_arch=model_id, stride=stride, k_exc=k_exc,
                                      simple_channels=simple_channels, complex_channels=complex_channels,
                                      noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level)

        if model_arch.lower() == 'resnet50_at':
            ckpt_data['state_dict'].pop('vone_block.div_u.weight')
            ckpt_data['state_dict'].pop('vone_block.div_t.weight')
            model.load_state_dict(ckpt_data['state_dict'])
        else:
            model = Wrapper(model)
            model.load_state_dict(ckpt_data['state_dict'])
            model = model.module

        model = nn.DataParallel(model)
    else:
        model = globals()[f'VOneNet'](model_arch=model_arch)
        model = nn.DataParallel(model)

    model.to("cpu")
    return model

def get_model(name):
    assert name == 'voneresnet-50-non_stochastic'
    model = get_model_from_s3()
    model = model.module
    preprocessing = functools.partial(load_preprocess_images, image_size=224,
                                      normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    wrapper = PytorchWrapper(identifier='vone' + 'resnet50_ns', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'voneresnet-50-non_stochastic'
    layers = (
            ['vone_block'] +
            ['model.layer1.0', 'model.layer1.1', 'model.layer1.2'] +
            ['model.layer2.0', 'model.layer2.1', 'model.layer2.2', 'model.layer2.3'] +
            ['model.layer3.0', 'model.layer3.1', 'model.layer3.2', 'model.layer3.3',
             'model.layer3.4', 'model.layer3.5'] +
            ['model.layer4.0', 'model.layer4.1', 'model.layer4.2'] +
            ['model.avgpool']
    )
    return layers


def get_bibtex(model_identifier):
    return """@inproceedings{NEURIPS2020_98b17f06,
 author = {Dapello, Joel and Marques, Tiago and Schrimpf, Martin and Geiger, Franziska and Cox, David and DiCarlo, James J},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
 pages = {13073--13087},
 publisher = {Curran Associates, Inc.},
 title = {Simulating a Primary Visual Cortex at the Front of CNNs Improves Robustness to Image Perturbations},
 url = {https://proceedings.neurips.cc/paper_files/paper/2020/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf},
 volume = {33},
 year = {2020}
}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
