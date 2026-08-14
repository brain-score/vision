import functools

import timm

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'seresnext50_32x4d'
# racm_in1k is the stronger of the two available tags (ImageNet top-1 ~81.3 vs
# ~79.9 for gluon_in1k).
TIMM_NAME = 'seresnext50_32x4d.racm_in1k'
IMAGE_SIZE = 224


def get_model(name):
    assert name == IDENTIFIER
    model = timm.create_model(TIMM_NAME, pretrained=True)
    model.eval()
    # timm reports mean=(0.485, 0.456, 0.406) std=(0.229, 0.224, 0.225) for this
    # tag, which is what load_preprocess_images already applies by default.
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    # The four residual stages plus the stem, and the classifier for the
    # behavioural readout.
    return ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']


def get_bibtex(model_identifier):
    return """@inproceedings{hu2018squeeze,
    title = {Squeeze-and-Excitation Networks},
    author = {Hu, Jie and Shen, Li and Sun, Gang},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2018},
}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
