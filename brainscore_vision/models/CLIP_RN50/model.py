from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from .helpers.clip_helpers import ClipModel
from brainscore_vision.model_helpers.check_submission import check_models



def get_model(name):
    assert name == "CLIP-RN50"
    clip_model = ClipModel(name.strip("CLIP-"))
    wrapper = PytorchWrapper(
        identifier=name,
        model=clip_model,
        preprocessing=clip_model.preprocessing
    )
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == "CLIP-RN50"
    layers = [
        'clmodel.visual.layer2.0.relu1',
        'clmodel.visual.layer2.1.relu1',
        'clmodel.visual.layer2.2.relu1',
        'clmodel.visual.layer2.3.relu1',
        'clmodel.visual.layer3.0.relu1',
        'clmodel.visual.layer3.1.relu1',
        'clmodel.visual.layer3.2.relu1',
        'clmodel.visual.layer3.3.relu1',
        'clmodel.visual.layer3.4.relu1',
        'clmodel.visual.layer4.0.relu1',
    ]
    return layers


def get_bibtex(model_identifier):
    return """
@article{radford2021learning,
title={Learning transferable visual models from natural language supervision},
author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
journal={arXiv preprint arXiv:2103.00020},
year={2021}
}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
