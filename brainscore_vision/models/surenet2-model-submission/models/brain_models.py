import functools
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_images
from model_tools.brain_transformation import ModelCommitment
from brainscore import score_model
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import sys


mydir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(mydir)
from surenet import SURENetHierarchy, URENetHierarchyEncode  # noqa: E402


model_name = "surenet-stage2-channel30,30-stride1,1-deg20"
model_file = "net5_channel30,30_stride1,1_noise0.15.pt"


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
    return [model_name]


def get_model(name):
    assert name == model_name
    net = torch.load(os.path.join(mydir, model_file))
    layer = URENetHierarchyEncode(net, i_layer=1)

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = layer

        def forward(self, *args, **kwargs):
            self.layer(*args, **kwargs)

    model = MyModel()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=model_name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    model = ModelCommitment(identifier=model_name, activations_model=wrapper, layers=["layer"],
                            region_layer_map={"V1": "layer"}, visual_degrees=20)
    return model


def get_layers(name):
    assert name == model_name
    return ["layer"]


def get_bibtex(model_identifier):
    return ""


if __name__ == '__main__':
    score = score_model(model_identifier=model_name, model=get_model(model_name),
                        benchmark_identifier="movshon.FreemanZiemba2013public.V1-pls")
    print(score)
