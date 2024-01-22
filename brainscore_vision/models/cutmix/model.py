# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from brainscore_vision.model_helpers.check_submission import check_models
import torch
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torchvision.models as models
import os
from torch.nn import Module
from brainscore_vision.model_helpers import download_weights

BIBTEX = """"""
LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']

models_folder_filename_version_sha = {
    "r50-e10-cut1": (
        "models/r50-epoch10-cutmix1",
        "model_best.pth.tar",
        "rCORHVh53BBzNpmHPgE8KhPiJzjPUqrZ",
        "c46ffef47eb96699c7d14de726d68228826c2378",
    ),
    "r50-e20-cut1": (
        "models/r50-epoch20-cutmix1",
        "model_best.pth.tar",
        "d8ocRsD_O5jol9VlwT253JZSyKZUf6qz",
        "7eb0de0fd2842e495f2b6d57172baafbc5f5ab61",
    ),
    "r50-e35-cut1": (
        "models/r50-epoch35-cutmix1",
        "model_best.pth.tar",
        "41xegwqEQ2Rg7.IRip.JdmfHP.2vOLai",
        "1dd17507d7cb7235832b8e9a8086cff90299dbca",
    ),
    "r50-e50-cut1": (
        'models/r50-e50-cutmix1',
        'model_best.pth.tar',
        'KvsE51zl2IM2lCKnmWMorOmSc8FTD1Gg',
        'b6c5bec71c4bc9740a9c8848d8c62963143dd995',
    ),
    "imgnfull-e45-cut1": (
        "models/epoch45-cutmix1",
        "model_best.pth.tar",
        "JIjF.ob9oWYP6EtAcuukix6DvawMj33B",
        "c08b69ab46165236606286a46afcaaea7b44df43",
    ),
    "imgnfull-e60-cut1": (
        "models/epoch60-cutmix1",
        "model_best.pth.tar",
        "pDsS2Egk_X19mX3XVc06NoOmTlzh1pJh",
        "de9393c1b1fc777043535571103efc4c99a6c3e7",
    ),
}


class Wrapper(Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model


def load_model(identifier, modelname='resnet', checkpoint_file=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    folder_path, filename, version, sha = models_folder_filename_version_sha[identifier]
    if modelname == 'resnet':
        model_type = models.resnet18 if "imgnfull" in identifier else models.resnet50
        model = model_type().to(device)
    else:
        raise ValueError("Architechture {} not valid.".format(modelname))
    print("=> loading checkpoint '{}'".format(checkpoint_file))

    save_directory = os.path.dirname(checkpoint_file)
    os.makedirs(save_directory, exist_ok=True)

    download_weights(
        bucket='brainscore-vision',
        folder_path=folder_path,
        filename_version_sha=[(
            filename,
            version,
            sha
        )],
        save_directory=save_directory)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model(identifier):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param identifier: the identifier of the model to fetch
    :return: the model instance
    """
    assert identifier in models_folder_filename_version_sha

    # init the model and the preprocessing:
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    dirname = os.path.dirname(__file__)
    weights_path = os.path.join(
        dirname, f"saved-weights/{identifier}/model_best.pth.tar")
    print(f"weights path is: {weights_path}")

    model = load_model(identifier=identifier, checkpoint_file=weights_path)

    # get an activations model from the Pytorch Wrapper
    activations_model = PytorchWrapper(
        identifier=identifier,
        model=model,
        preprocessing=preprocessing)

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


# Main Method: In submitting a custom model, you should not have to mess
# with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
