import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.s3 import load_weight_file
import torch
from .evnet.evnet import EVNet, evnet_params

def get_model(name):
    assert name == 'voneresnet_50_3'
    model = EVNet(
        **evnet_params['vonenet'], model_arch='resnet50', image_size=224,
        gabor_seed=1, visual_degrees=7, num_classes=1000,
        sf_max=11.5, k_exc=25, stride=4, ksize=31
        )
    weight_file = load_weight_file(
        bucket="evnets-model-weights",
        relative_path="voneresnet_50_1.pth",
        sha1="608096f8f2976836abe5481c85438210b0c6f46d",
        version_id="null"
        )
    model.to(torch.device('cpu'))
    checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    preprocessing = functools.partial(
        load_preprocess_images,
        image_size=224,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5)
        )
    wrapper = PytorchWrapper(
        identifier='voneresnet_50_3',
        model=model, preprocessing=preprocessing
        )
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'voneresnet_50_3'
    layers = (
            ['voneblock'] +
            ['model.0.0', 'model.0.1', 'model.0.2'] +
            ['model.1.0', 'model.1.1', 'model.1.2', 'model.1.3'] +
            ['model.2.0', 'model.2.1', 'model.2.2', 'model.2.3',
             'model.2.4', 'model.2.5'] +
            ['model.3.0', 'model.3.1', 'model.3.2'] +
            ['model.4', 'model.5']
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
