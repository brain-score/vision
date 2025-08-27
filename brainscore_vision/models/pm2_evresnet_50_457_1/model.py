import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
from .evnet.evnet import EVNet

def get_model(name):
    assert name == 'pm2_evresnet_50_457_1'
    model = EVNet(
        with_retinablock=True, with_voneblock=True, model_arch='resnet50',
        image_size=224, visual_degrees=7, num_classes=1000,
        colors_p_cells=['r/g', 'g/r', 'b/y'], p_channels=3, m_channels=1, retinal_noise_mode=None,
        noise_mode=None, gabor_seed=457, sf_max=9, stride=4, gabor_color_prob=[.25, .25,.25, .25], k_exc=3.9,
        light_adapt_mode='weber'
        )
    model.to(torch.device('cpu'))
    preprocessing = functools.partial(
        load_preprocess_images,
        image_size=224,
        normalize_mean=(0, 0, 0),
        normalize_std=(1, 1, 1)
        )
    wrapper = PytorchWrapper(
        identifier='pm2_evresnet_50_457_1',
        model=model, preprocessing=preprocessing
        )
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'pm2_evresnet_50_457_1'
    layers = (
            ['voneblock'] +
            ['model.layer1.0', 'model.layer1.1', 'model.layer1.2'] +
            ['model.layer2.0', 'model.layer2.1', 'model.layer2.2', 'model.layer2.3'] +
            ['model.layer3.0', 'model.layer3.1', 'model.layer3.2', 'model.layer3.3',
             'model.layer3.4', 'model.layer3.5'] +
            ['model.layer4.0', 'model.layer4.1', 'model.layer4.2'] +
            ['model.avgpool']
    )
    return layers


def get_bibtex(model_identifier):
    return """@misc{piper2025explicitlymodelingsubcorticalvision,
      title={Explicitly Modeling Subcortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness}, 
      author={Lucas Piper and Arlindo L. Oliveira and Tiago Marques},
      year={2025},
      eprint={2506.03089},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.03089}
      }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
