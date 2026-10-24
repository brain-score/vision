"""alexnet_altahan_03: AlexNet with local response normalisation, fine-tuned on ImageNet with a SimCLR auxiliary,
framed, edge-map and wide-crop training views, and behaviour read from centred, L2-normalised logits.

torchvision AlexNet with the original AlexNet's parameter-free local response normalisation restored after the conv1
and conv2 ReLUs, fine-tuned on ImageNet-1k (SGD, lr 1e-3, momentum 0.9, weight decay 5e-4, batch 256, 5.12M images;
cross-entropy plus a SimCLR NT-Xent auxiliary loss on fc7 through a projection head that is discarded after training).
Training views: random-resized-crop with area scale 0.35-1 and horizontal flip; for the cross-entropy input, 10% of
the images are replaced by their Sobel edge map, and 80% are then shrunk by a factor U[0.35, 1] and pasted centred on
a uniform grey canvas of random level; the SimCLR views are unframed. The LRN layers have no weights, so the network
has AlexNet's 61,100,840 parameters and the state dict is torchvision's. Read out with Brain-Score's committed
'alexnet' layer map except V4, which is read from conv4 (features.9); behaviour from the 1000 logits, centred and
L2-normalised per image by the parameter-free `logits_centred_l2` module (registered before the classifier, so
Brain-Score's default `logits` layer is still classifier.6, the output of forward()).
"""
import functools

import torch
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'alexnet_altahan_03'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/alexnet_altahan_03.pt'
WEIGHTS_VERSION_ID = 'lqPOrYaBymcko0r2GCxQlUHhi1LPK5Q6'
WEIGHTS_SHA1 = '28224a6388dd1488a08dbb99690ec47ed3379cb8'   # checked after download (load_file does not verify it)

BIBTEX = """@incollection{NIPS2012_4824,
                  title = {ImageNet Classification with Deep Convolutional Neural Networks},
                  author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                  booktitle = {Advances in Neural Information Processing Systems 25},
                  pages = {1097--1105},
                  year = {2012},
                  publisher = {Curran Associates, Inc.}
                  }"""

IMAGE_SIZE = 224
LAYERS = ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',
          'classifier.2', 'classifier.5', 'classifier.6', 'logits_centred_l2']
# Brain-Score's committed map for its 'alexnet' plugin (region_layer_map/alexnet.json).
# Set to None to let Brain-Score's LayerSelection choose layers on the public dev benchmarks.
REGION_LAYER_MAP = {'V1': 'features.2', 'V2': 'features.7', 'V4': 'features.9', 'IT': 'features.12'}
BEHAVIORAL_READOUT_LAYER = 'logits_centred_l2'   # the 1000 logits, centred and L2-normalised per image (parent: 'classifier.6')
VISUAL_DEGREES = 8            # how large the model's field of view is taken to be (Brain-Score's default)


class CentredL2(torch.nn.Module):
    """Per-image (l - mean(l)) / ||l - mean(l)||_2 over the last dim. No parameters."""

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class AlexNetWithLogitTap(torch.nn.Module):
    """AlexNet whose forward() also passes its logits through the `logits_centred_l2` tap (output discarded;
    Brain-Score reads it via a hook) and returns the plain logits. The tap is registered first, so
    classifier.6 stays the last-registered module, and module names and state-dict keys match AlexNet's."""

    def __init__(self, base):
        super().__init__()
        self.logits_centred_l2 = CentredL2()
        self.features, self.avgpool, self.classifier = base.features, base.avgpool, base.classifier

    def forward(self, x):
        logits = self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))
        self.logits_centred_l2(logits)
        return logits


def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    path = load_file(bucket='brainscore-storage', folder_name='brainscore-vision/models',
                     relative_path=WEIGHTS_RELATIVE_PATH, version_id=WEIGHTS_VERSION_ID)
    if sha1_hash(path) != WEIGHTS_SHA1:
        raise ValueError(f'{path} does not match sha1 {WEIGHTS_SHA1}')
    model = torchvision.models.alexnet(weights=None)
    # Original AlexNet's parameter-free local response normalisation after the conv1 and conv2 ReLUs. Each ReLU
    # becomes Sequential(ReLU, LRN) in place, so features.2/5/... keep their names and the state dict is unchanged.
    for i in (1, 4):
        model.features[i] = torch.nn.Sequential(model.features[i],
                                                torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2))
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model = AlexNetWithLogitTap(model).eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    return LAYERS


def get_bibtex(model_identifier):
    return BIBTEX


if __name__ == '__main__':
    check_models.check_base_models(__name__)
