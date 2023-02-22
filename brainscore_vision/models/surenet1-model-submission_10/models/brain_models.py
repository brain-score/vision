import os
import sys
mydir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(mydir)


import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.check_submission import check_models
from model_tools.activations.pytorch import load_images
from model_tools.activations.pytorch import PytorchWrapper
import functools
from surenet import SURENetHierarchy, URENetHierarchyEncode


def torchvision_preprocess(normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        transforms.Grayscale(),
        lambda img: img.unsqueeze(0)
    ])


def torchvision_preprocess_input(image_size, **kwargs):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        torchvision_preprocess(**kwargs),
    ])


def preprocess_images(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images


def load_preprocess_images(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images


def get_model_list():
    return ["surenet-1"]


def get_model(name):
    assert name == "surenet-1"
    net = torch.load(os.path.join(mydir, "net.pt"))
    layer = URENetHierarchyEncode(net, i_layer=0)

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = layer

        def forward(self, *args, **kwargs):
            self.layer(*args, **kwargs)

    model = MyModel()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier="surenet-1", model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    model = ModelCommitment(identifier="surenet-1", activations_model=wrapper, layers=["layer"],
                            region_layer_map={"V1": "layer"}, visual_degrees=8)
    return model


def get_layers(name):
    assert name == "surenet-1"
    return ["layer"]


def get_bibtex(model_identifier):
    return ""


if __name__ == '__main__':
    score = score_model(model_identifier="surenet-1", model=get_model("surenet-1"),
                        benchmark_identifier="movshon.FreemanZiemba2013public.V1-pls")
    print(score)
    # check_models.check_brain_models(__name__)
