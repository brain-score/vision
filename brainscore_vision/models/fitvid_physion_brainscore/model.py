import numpy as np
import functools
from torchvision import transforms as T
import torch
from torch import nn, Tensor
from typing import Dict, Iterable, Callable


from model_tools.activations.pytorch import PytorchWrapper, load_images
from model_tools.check_submission import check_models
from physion import load_fitvid


def get_model_list():
    return ["fitvid_trained_on_physion",
            "fitvid_random"]


def load_preprocess_images_to_videos(image_filepaths, image_size, video_length=5):
    """
    define custom pre-processing here since fitvid accepts videos
    """
    images = load_images(image_filepaths)
    # preprocessing
    transforms = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),  # ToTensor() divides by 255
        lambda img: img.repeat(video_length, 1, 1, 1),
        lambda img: img.unsqueeze(0),
    ])
    images = [transforms(img) for img in images ]
    images = np.concatenate(images)
    return images


class Slicer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x[0][:, 0]


class Postprocess(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self.slicers = torch.nn.ModuleDict({layer.replace('.', '__'): Slicer() for layer in layers})
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return {layer: self.slicers[layer.replace('.', '__')](self._features[layer]) for layer in self.layers}

    
def get_model(name):
    if name == "fitvid_trained_on_physion":
        model = load_fitvid() # pretrained
    else:
        assert name == "fitvid_random"
        model = load_fitvid(pretrained=False)
    ppmodel = Postprocess(model, ['encoder'])
    preprocessing = functools.partial(load_preprocess_images_to_videos, image_size=64)
    wrapper = PytorchWrapper(identifier=name, model=ppmodel, preprocessing=preprocessing)
    wrapper.image_size = 64
    return wrapper
    

def get_layers(name):
    if name == "fitvid_trained_on_physion":
        model = load_fitvid() # pretrained
    else:
        assert name == "fitvid_random"
        model = load_fitvid(pretrained=False)
    return ["slicers.encoder"]


def get_bibtex(model_identifier):
    return """@misc{2106.13195,
    Author = {Mohammad Babaeizadeh and Mohammad Taghi Saffar and Suraj Nair and Sergey Levine\
 and Chelsea Finn and Dumitru Erhan},
    Title = {FitVid: Overfitting in Pixel-Level Video Prediction},
    Year = {2021},
    Eprint = {arXiv:2106.13195}}"""

if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
