import functools

import torchvision.models
import torch 

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

BIBTEX = """@article{geirhos2018imagenet,
  title={ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness},
  author={Geirhos, Robert and Rubisch, Patricia and Michaelis, Claudio and Bethge, Matthias and Wichmann, Felix A and Brendel, Wieland},
  journal={arXiv preprint arXiv:1811.12231},
  year={2018}
}"""

LAYERS = ['features.module.2', 'features.module.5', 'features.module.7', 'features.module.9', 'features.module.12',
          'classifier.2', 'classifier.5']

model_url =  'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar'

def get_model():
    model = torchvision.models.alexnet(pretrained=False)
    model.features = torch.nn.DataParallel(model.features)
    checkpoint = torch.utils.model_zoo.load_url(model_url, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='AlexNet_SIN', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper
