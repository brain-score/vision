import functools
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import torchvision.models

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_model(name):
    assert name == 'efficientnet_b0'
    model = torchvision.models.efficientnet_b0(pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='efficientnet_b0', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'efficientnet_b0'
    return ['features.0.2',
             'features.2.1.stochastic_depth',
             'features.3.1.stochastic_depth',
             'features.4.1.stochastic_depth',
             'features.4.2.stochastic_depth',
             'features.5.1.stochastic_depth',
             'features.5.2.stochastic_depth',
             'features.6.1.stochastic_depth',
             'features.6.2.stochastic_depth',
             'features.8.2','classifier.0','classifier.1',
             ]


def get_bibtex(model_identifier):
    return """@misc{tan2020efficientnet,
      title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks}, 
      author={Mingxing Tan and Quoc V. Le},
      year={2020},
      eprint={1905.11946},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)