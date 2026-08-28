"""Bao et al. 2020 object space: AlexNet fc6 restricted to its top 50 principal components.

The paper builds the space from "the responses of 4,096 nodes in layer fc6" of pretrained AlexNet,
keeps "the first 50 PCs, which captured 85% of the response variance", and fits single IT neurons
as R = c.F + c0 over those 50 dimensions.

Deviations from the paper:
- The PCA is fit on the 1,854 THINGS object images (Hebart2023). The paper fits on its own 1,224
  stimuli (51 objects x 24 views), which are not released.
- fc6 is read pre-ReLU. The paper does not specify, but MatConvNet -- the AlexNet for MATLAB, which
  the paper's analysis code uses -- stores fc6 and relu6 as separate layers.
- fc6 is reconstructed from the top 50 PCs rather than emitted as a 50-vector; the two are
  equivalent under the standard PLSRegression(n_components=25) readout.

All regions are committed to the object space, so non-IT scores act as a control.
"""

import os
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from PIL import Image
from result_caching import store_dict
from sklearn.decomposition import PCA

from brainscore_vision.model_helpers.activations.core import change_dict, flatten
from brainscore_vision.model_helpers.activations.pca import LayerPCA
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

BIBTEX = """@article{bao2020map,
                  title = {A map of object space in primate inferotemporal cortex},
                  author = {Bao, Pinglei and She, Liang and McGill, Mason and Tsao, Doris Y.},
                  journal = {Nature},
                  volume = {583},
                  number = {7814},
                  pages = {103--108},
                  year = {2020},
                  publisher = {Nature Publishing Group},
                  doi = {10.1038/s41586-020-2350-5}
                  }"""

IDENTIFIER = 'bao2020_objectspace_50d'
LAYER = 'fc6'
N_COMPONENTS = 50
FIT_STIMULI = 'Hebart2023'
IMAGE_SIZE = 227
MATCONVNET_URL = 'https://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat'
LAYERS = [LAYER]
REGION_LAYER_MAP = {region: LAYER for region in ('V1', 'V2', 'V4', 'IT')}


class CaffeAlexNet(nn.Module):
    """Krizhevsky 2012 topology, as distributed in MatConvNet's imagenet-caffe-alex."""

    def __init__(self):
        super().__init__()
        lrn = dict(size=5, alpha=1e-4, beta=0.75, k=1.0)  # matconvnet [5, 1, 2e-5, .75]; alpha *= size
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4), nn.ReLU(inplace=False),
            nn.LocalResponseNorm(**lrn), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2, groups=2), nn.ReLU(inplace=False),
            nn.LocalResponseNorm(**lrn), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(384, 384, 3, padding=1, groups=2), nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, 3, padding=1, groups=2), nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2),
        )
        self.fc6 = nn.Linear(9216, 4096)
        self.relu6 = nn.ReLU(inplace=False)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=False)
        self.fc8 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc8(self.relu7(self.fc7(self.relu6(self.fc6(x)))))


def _convert_matconvnet(mat_path):
    mat = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    weights = {layer.name: [np.asarray(w) for w in layer.weights]
               for layer in mat['layers'] if getattr(layer, 'type', '') == 'conv'}
    net = CaffeAlexNet()
    convs = [('conv1', 0), ('conv2', 4), ('conv3', 8), ('conv4', 10), ('conv5', 12)]
    for name, index in convs:
        w, b = weights[name]
        net.features[index].weight.data = torch.from_numpy(w.transpose(3, 2, 0, 1).copy()).float()
        net.features[index].bias.data = torch.from_numpy(b.copy()).float()
    w, b = weights['fc6']  # (6, 6, 256, 4096) -> (4096, 256*6*6) in NCHW flattening order
    net.fc6.weight.data = torch.from_numpy(w.transpose(3, 2, 0, 1).reshape(4096, -1).copy()).float()
    net.fc6.bias.data = torch.from_numpy(b.copy()).float()
    for name, module in [('fc7', net.fc7), ('fc8', net.fc8)]:
        w, b = weights[name]
        module.weight.data = torch.from_numpy(np.atleast_2d(w).T.copy()).float()
        module.bias.data = torch.from_numpy(b.copy()).float()
    average_image = np.asarray(mat['meta'].normalization.averageImage, dtype=np.float32)
    return {'state_dict': net.state_dict(), 'average_image': average_image}


def _load_weights():
    cache_dir = Path(torch.hub.get_dir()) / 'checkpoints'
    cache_dir.mkdir(parents=True, exist_ok=True)
    converted = cache_dir / 'bvlc_alexnet_matconvnet.pth'
    if not converted.exists():
        mat_path = cache_dir / 'imagenet-caffe-alex.mat'
        if not mat_path.exists():
            torch.hub.download_url_to_file(MATCONVNET_URL, str(mat_path))
        torch.save(_convert_matconvnet(mat_path), str(converted))
    return torch.load(str(converted), weights_only=False)


class ObjectSpacePCA(LayerPCA):
    """Restricts `LAYER` to its top PCs; every other layer passes through untouched."""

    def __call__(self, batch_activations):
        self._ensure_initialized(batch_activations.keys())

        def restrict(layer, activations):
            pca = self._layer_pcas[layer]
            activations = flatten(activations)
            return activations if pca is None else pca.inverse_transform(pca.transform(activations))

        return change_dict(batch_activations, restrict, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _pcas(self, identifier, layers, n_components):
        if LAYER not in layers:
            return {layer: None for layer in layers}
        from brainscore_vision import load_stimulus_set
        self.handle.disable()
        activations = self._extractor(load_stimulus_set(FIT_STIMULI), layers=[LAYER])
        self.handle.enable()
        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(flatten(activations.sel(layer=LAYER).values))
        return {layer: pca if layer == LAYER else None for layer in layers}


def get_model():
    blob = _load_weights()
    model = CaffeAlexNet()
    model.load_state_dict(blob['state_dict'])
    model.eval()
    average_image = blob['average_image']

    def preprocessing(image_filepaths):  # caffe-style: 0-255 RGB minus the per-pixel mean image
        images = [np.asarray(Image.open(path).convert('RGB').resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC), dtype=np.float32) - average_image
            for path in image_filepaths]
        return np.stack(images).transpose(0, 3, 1, 2)

    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    pca = ObjectSpacePCA(wrapper, n_components=N_COMPONENTS)  # LayerPCA.hook hardcodes LayerPCA
    pca.handle = wrapper.register_batch_activations_hook(pca)
    return wrapper
