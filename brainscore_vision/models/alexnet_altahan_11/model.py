"""alexnet_altahan_11: AlexNet fine-tuned on ImageNet with a cosine-softmax classifier, SAM and shape-cue augmentations.

torchvision AlexNet, fine-tuned from the ImageNet-1k weights on ImageNet-1k for 5.12M images (SAM, rho 0.05, around SGD
with lr 1e-3 on a cosine schedule, momentum 0.9, weight decay 5e-4, batch 256; random-resized-crop and horizontal flip).
classifier.6 is replaced by cosine-softmax logits s * cos(fc7, w_c): no bias, w_c initialised from the pretrained
classifier.6 weights, and a learnable scale s initialised to 92.5. Augmentation: a random 15% of the images are replaced
by their grayscale Canny edge map (black edges on white); a random half of the others get Gaussian blur (sigma up to
3 px) and grayscale uniform noise; the remaining clean images get APR-S amplitude/phase recombination (20%), contrast
reduction toward the mean luminance (25%) and AugMix; then bar and rectangle occluders (uniform gray, black or white,
10-40% of the image) on 30% of the non-edge images. AugMix, APR-S and contrast reduction are switched off for the last
10% of the images. Read out with Brain-Score's committed 'alexnet' layer map (V1 features.2, V2 and V4 features.7, IT
features.12) at a 3-degree visual field; behaviour from a parameter-free softmax (T=1) of the logits ('logit_softmax');
the logits are classifier.6, the output of forward().
"""
import functools

import torch
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'alexnet_altahan_11'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/alexnet_altahan_11.pt'
WEIGHTS_VERSION_ID = 'aa836ud0YHObLBNzjraIBoiFeJiHG1p9'
WEIGHTS_SHA1 = 'dde7a58f54a83e42cbf38710d5d02c9243130d15'   # checked after download (load_file does not verify it)

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
          'classifier.2', 'classifier.5', 'logit_softmax']
# Brain-Score's committed map for its 'alexnet' plugin (region_layer_map/alexnet.json).
# Set to None to let Brain-Score's LayerSelection choose layers on the public dev benchmarks.
REGION_LAYER_MAP = {'V1': 'features.2', 'V2': 'features.7', 'V4': 'features.7', 'IT': 'features.12'}
# Behavior reads a fixed softmax (T=1) of the logits: non-negative class probabilities, nothing fit.
BEHAVIORAL_READOUT_LAYER = 'logit_softmax'
VISUAL_DEGREES = 3            # how large the model's field of view is taken to be (Brain-Score's default is 8)
# Initial cosine-logit scale: matches the pretrained classifier.6's logits on ImageNet train images (mean per-image
# range 25.5 vs cosine range 0.275, per-image std 3.64 vs 0.0394, both ratios ~92.5; mean max logit ~16.5).
COSINE_SCALE_INIT = 92.5


class CosineLinear(torch.nn.Module):
    """Cosine-softmax logits s * cos(x, w_c) (no bias), with a learnable scale s. Computed in float32 under autocast:
    cosines span only ~0.3, so half-precision rounding would be amplified by s."""
    def __init__(self, weight, scale=COSINE_SCALE_INIT):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().clone())
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = torch.nn.functional.normalize(x.float(), dim=1)
            w = torch.nn.functional.normalize(self.weight.float(), dim=1)
            return self.scale.float() * torch.nn.functional.linear(x, w)


class SoftmaxTapAlexNet(torch.nn.Module):
    """AlexNet that also passes its logits through a parameter-free softmax tap ('logit_softmax').
    Same state-dict keys as torchvision's AlexNet, and forward() still returns the raw logits. The tap is
    registered first, so classifier.6 (the logits) stays the last-registered module."""
    def __init__(self, base):
        super().__init__()
        self.logit_softmax = torch.nn.Softmax(dim=1)
        self.features, self.avgpool, self.classifier = base.features, base.avgpool, base.classifier

    def forward(self, x):
        logits = self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))
        self.logit_softmax(logits)   # read out by Brain-Score's hook only
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
    model.classifier[6] = CosineLinear(model.classifier[6].weight)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model = SoftmaxTapAlexNet(model).eval()
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
