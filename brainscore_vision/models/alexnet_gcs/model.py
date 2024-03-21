import functools

import torchvision.models
import torch
from torchvision.transforms import ToTensor
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from .GDS import ToRetinalGanglionCellSampling
import numpy as np

BIBTEX = """@incollection{NIPS2012_4824,
                  title = {ImageNet Classification with Deep Convolutional Neural Networks},
                  author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                  booktitle = {Advances in Neural Information Processing Systems 25},
                  editor = {F. Pereira and C. J. C. Burges and L. Bottou and K. Q. Weinberger},
                  pages = {1097--1105},
                  year = {2012},
                  publisher = {Curran Associates, Inc.},
                  url = {http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf}
                  }"""

GCS_LAYERS = ['model.features.2', 'model.features.5', 'model.features.7', 'model.features.9', 'model.features.12',
          'model.classifier.2', 'model.classifier.5']
LAYERS = ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',
          'classifier.2', 'classifier.5']
IMAGE_SIZE = 224

class AlexnetGCS(torchvision.models.AlexNet):
    def __init__(self, model, fov=20):
        super().__init__()
        self.model = model

        self.gcs = ToRetinalGanglionCellSampling(fov=fov, image_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), out_size=IMAGE_SIZE, series=1, dtype=np.float32)

    def forward(self, x):
        batch = []
        for _x in x:
            _x = _x.permute(1, 2, 0)
            _x = _x.cpu().numpy()
            _x = self.gcs(_x)
            _x = ToTensor()(_x)
            # _x = _x.permute(1, 2, 0)
            batch.append(_x)
        x = torch.stack(batch)
        return self.model.forward(x)

def get_gcs_model(fov=20):
    model = torchvision.models.alexnet(pretrained=True)
    model = AlexnetGCS(model, fov=fov)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=f'alexnet_gcs_FOV-{fov}', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_model():
    model = torchvision.models.alexnet(pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='alexnet', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper
