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
    return ["fitvid_trained_on_physion_alllayers",
            "fitvid_random_alllayers"]


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


class TupleSlicer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x[0][:, 0]
    

class SliceReshaper(nn.Module):
    def __init__(self, video_length):
        super().__init__()
        self.video_length = video_length
        
    def forward(self, x):
        y = x[::self.video_length]
        if y.shape[-2:] == (1, 1):
            y = y[:, :, 0, 0]
        return y
    
    
class Postprocess(nn.Module):
    def __init__(self,
                 model,
                 layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self.slicers = {}
        for layer in layers:
            klass = layers[layer]['class']
            args = layers[layer].get('args', ())
            kwargs = layers[layer].get('kwargs', {})
            self.slicers[layer.replace('.', '__')] = klass(*args, **kwargs)
        self.slicers = torch.nn.ModuleDict(self.slicers)
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


def get_layerdict():
    llist = ['conv1', 'conv2',  'se.squeeze', 'se.expand']
    layerdict = {'encoder.blocks.%d.%d.%s' % (x, y, z): {'class': SliceReshaper, 
                                                         'kwargs': {'video_length': 5}} 
                 for x in [0, 1, 2, 3] for y in [0,1] for z in llist}
    layerdict['encoder'] = {'class': TupleSlicer}
    return layerdict

    
def get_model(name):
    if name == "fitvid_trained_on_physion_alllayers":
        model = load_fitvid() # pretrained
    else:
        assert name == "fitvid_random_alllayers"
        model = load_fitvid(pretrained=False)

    layerdict = get_layerdict()
    ppmodel = Postprocess(model, layerdict)
    preprocessing = functools.partial(load_preprocess_images_to_videos, image_size=64)
    wrapper = PytorchWrapper(identifier=name, model=ppmodel, preprocessing=preprocessing)
    wrapper.image_size = 64
    return wrapper
    

def get_layers(name):
    if name == "fitvid_trained_on_physion_alllayers":
        model = load_fitvid() # pretrained
    else:
        assert name == "fitvid_random_alllayers"
        model = load_fitvid(pretrained=False)

    layerdict = get_layerdict()        
    layerlist = ["slicers.%s" % (s.replace('.', '__')) for s in layerdict]
    return layerlist

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
