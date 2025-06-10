import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.s3 import load_weight_file
import torch
from .evnet.evnet import EVNet

def get_model(name):
    assert name == 'voneresnet_50_457_1'
    model = EVNet(
        with_retinablock=False, with_voneblock=True, model_arch='resnet50',
        image_size=224, visual_degrees=7, num_classes=1000,
        colors_p_cells=['r/g', 'g/r', 'b/y'], p_channels=3, m_channels=0, retinal_noise_mode=None,
        noise_mode=None, gabor_seed=457, sf_max=9, stride=4, gabor_color_prob=None
        )
    model.to(torch.device('cpu'))
    preprocessing = functools.partial(
        load_preprocess_images,
        image_size=224,
        normalize_mean=(.5, .5, .5),
        normalize_std=(.5, .5, .5)
        )
    wrapper = PytorchWrapper(
        identifier='voneresnet_50_457_1',
        model=model, preprocessing=preprocessing
        )
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'voneresnet_50_457_1'
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
    return """@misc{piper2024explicitlymodelingprecorticalvision,
      title={Explicitly Modeling Pre-Cortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness}, 
      author={Lucas Piper and Arlindo L. Oliveira and Tiago Marques},
      year={2024},
      eprint={2409.16838},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.16838}
      }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
