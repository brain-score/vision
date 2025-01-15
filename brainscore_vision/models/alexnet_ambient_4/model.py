
import functools
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
import gdown
from torch import nn
import torchvision.models as models


def get_bibtex(model_identifier):
    return ""

def get_model_list(network, keyword, iteration):
    return [f"alexnet_ambient_iteration=4"]

def download_checkpoint(network, keyword, iteration):
    url = f"https://eggerbernhard.ch/shreya/latest_alexnet/ambient_4.ckpt"
    output = f"alexnet_ambient_iteration=4.ckpt"
    gdown.download(url, output, quiet=False)
    return output

def load_checkpoint(model, checkpoint_path, network, keyword):
    if checkpoint_path == "x" or keyword == "imagenet_trained":
        return model
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt2 = {}

    for keys in ckpt["state_dict"]:
        k2 = keys.split("model.")[1]
        ckpt2[k2] = ckpt["state_dict"][keys]
    model.load_state_dict(ckpt2)
    return model

def get_model(network, keyword, iteration):
    checkpoint_path = download_checkpoint(network, keyword, iteration)
    pretrained = keyword == "imagenet_trained"
    
    model = torch.hub.load('pytorch/vision', network, pretrained=pretrained)

    if not pretrained:
        model = load_checkpoint(model, checkpoint_path, network, keyword)

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=f"alexnet_ambient_iteration=4",
                                       model=model,
                                       preprocessing=preprocessing)
    return activations_model

def get_layers(network, keyword, iteration):
    checkpoint_path = download_checkpoint(network, keyword, iteration)
    pretrained = keyword == "imagenet_trained"

    model = torch.hub.load('pytorch/vision', network, pretrained=pretrained)

    if not pretrained:
        model = load_checkpoint(model, checkpoint_path, network, keyword)

    layers = [name for name, module in model.named_modules()]
    return layers

if __name__ == '__main__':
    device = "cpu"
    network = f"alexnet"  # Example network
    keyword = f"ambient"  # Example keyword
    iteration = f"4"  # Example iteration

    url = f"https://eggerbernhard.ch/shreya/latest_alexnet/ambient_4.ckpt"
    output = f"alexnet_ambient_iteration=4.ckpt"
    gdown.download(url, output)

    model = get_model(network, keyword, iteration)
    layers = get_layers(network, keyword, iteration)
    print(f"Loaded model:", model)
    print(f"Available layers:", layers)
