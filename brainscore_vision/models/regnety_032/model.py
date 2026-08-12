import functools

import timm

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'regnety_032'
# ra_in1k is the RandAugment recipe (ImageNet top-1 ~82.0), the strongest of the
# three available regnety_032 tags.
TIMM_NAME = 'regnety_032.ra_in1k'
IMAGE_SIZE = 224


def get_model(name):
    assert name == IDENTIFIER
    model = timm.create_model(TIMM_NAME, pretrained=True)
    model.eval()
    # ra_in1k expects standard ImageNet normalisation, which is what
    # load_preprocess_images already applies by default.
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    # The four RegNet stages plus the stem, and the classifier head for the
    # behavioural readout.
    return ['stem', 's1', 's2', 's3', 's4', 'final_conv', 'head']


def get_bibtex(model_identifier):
    return """@inproceedings{radosavovic2020designing,
    title = {Designing Network Design Spaces},
    author = {Radosavovic, Ilija and Kosaraju, Raj Prateek and Girshick, Ross and He, Kaiming and Doll{\\'a}r, Piotr},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2020},
}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
