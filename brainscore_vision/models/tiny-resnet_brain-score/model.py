from torchvision.transforms import Grayscale
from typing import List

from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from model_tools.activations import PytorchWrapper
from model_tools.activations.pytorch import load_images
from model_tools.check_submission import check_models

"""
Template module for a base model submission to brain-score
"""


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ["resnet_tiny-mnist"]


def get_model(name: str) -> PytorchWrapper:
    assert name == "resnet_tiny-mnist"
    preprocessing = AutoFeatureExtractor.from_pretrained("fxmarty/resnet-tiny-mnist")

    def load_and_preprocess(paths):
        images = load_images(paths)
        # images = [image.convert('L') for image in images]  # to grayscale
        images = [Grayscale(num_output_channels=3)(image) for image in images]
        model_inputs = preprocessing(images)['pixel_values']
        model_inputs = [image_rgb[:1] for image_rgb in model_inputs]  # all three channels are identical, just take one
        return model_inputs

    model = AutoModelForImageClassification.from_pretrained("fxmarty/resnet-tiny-mnist")
    image_model = model.resnet
    wrapper = PytorchWrapper(image_model, preprocessing=load_and_preprocess, identifier=name)
    return wrapper


def get_layers(name: str) -> List[str]:
    assert name == "resnet_tiny-mnist"
    return ['embedder'] + \
        [f'encoder.stages.{stage}.layers.{layer}' for stage, layers in enumerate([2, 2]) for layer in range(layers)] + \
        ['pooler']


def get_bibtex(model_identifier):
    return None


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
