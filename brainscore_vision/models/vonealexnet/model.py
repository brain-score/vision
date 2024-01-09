# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from .vonenet import get_model as get_voenet_model

"""
Template module for a base model submission to brain-score
"""

BIBTEX = ""


def get_layers(identifier):
    assert identifier in {"vone_alexnet", "vone_alexnet_full"}

    if identifier == "vone_alexnet":
        return [
            "module.vone_block.output",
            "module.model.features.1",
            "module.model.features.4",
            "module.model.features.6",
            "module.model.features.8",
            "module.model.classifier.2",
            "module.model.classifier.5",
            "module.model.classifier.6",
        ]
    else:
        model_arch = "alexnet"
        MyModel = get_voenet_model(model_arch=model_arch, pretrained=True)
        return [layer for layer, _ in MyModel.named_modules()][1:]


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model(identifier):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert identifier in {"vone_alexnet", "vone_alexnet_full"}
    # link the custom model to the wrapper object

    # define your custom model here:
    model_arch = "alexnet"
    MyModel = get_voenet_model(model_arch=model_arch, pretrained=True)

    # init the model and the preprocessing:
    preprocessing = functools.partial(load_preprocess_images, image_size=224)

    # get an activations model from the Pytorch Wrapper
    wrapper = PytorchWrapper(
        identifier=identifier, model=MyModel, preprocessing=preprocessing
    )
    wrapper.image_size = 224
    return wrapper


# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
