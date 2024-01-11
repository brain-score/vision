import functools
import torch
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

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
    return ["barlow-twins-resnet50"]


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == "barlow-twins-resnet50"
    model = torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


LAYERS = (["relu", "maxpool"]
        + ["layer1.0.relu", "layer1.1.relu"]
        + ["layer2.0.relu", "layer2.1.relu"]
        + ["layer3.0.relu", "layer3.1.relu"]
        + ["layer4.0.relu", "layer4.1.relu"]
    )


BIBTEX = """@misc{zbontar2021barlow,
    title={Barlow Twins: Self-Supervised Learning via Redundancy Reduction}, 
    author={Jure Zbontar and Li Jing and Ishan Misra and Yann LeCun and Stéphane Deny},
    year={2021},
    eprint={2103.03230},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }"""


if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
