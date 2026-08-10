import functools

from vonenet import get_model as get_vonenet

from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images, PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = "vonealexnet"

LAYERS = [
    "vone_block",
    "bottleneck",
    "model.features.2",
    "model.features.4",
    "model.features.6",
    "model.features.8",
    "model.avgpool",
    "model.classifier.2",
    "model.classifier.5",
]


def get_model(name: str) -> PytorchWrapper:
    assert name == IDENTIFIER
    model = get_vonenet("alexnet").module
    # ponytail: deterministic activations; drop this line to score the paper's stochastic model
    model.vone_block.set_noise_mode(None)
    preprocessing = functools.partial(
        load_preprocess_images,
        image_size=224,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
    )
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name: str) -> list:
    assert name == IDENTIFIER
    return LAYERS


def get_bibtex(name: str) -> str:
    return """@inproceedings{dapello2020simulating,
 author = {Dapello, Joel and Marques, Tiago and Schrimpf, Martin and Geiger, Franziska and Cox, David and DiCarlo, James J},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {13073--13087},
 title = {Simulating a Primary Visual Cortex at the Front of CNNs Improves Robustness to Image Perturbations},
 volume = {33},
 year = {2020}
}"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
