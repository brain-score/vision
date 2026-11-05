"""PytorchWrapper declares the layer paths it can record."""
import numpy as np
import torch
from PIL import Image

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper


def _wrapper():
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 3), torch.nn.Flatten()).eval()

    def preprocess(paths):
        return np.stack([np.asarray(Image.open(p).convert('RGB'), dtype=np.float32).transpose(2, 0, 1)
                         for p in paths])
    return PytorchWrapper(net, preprocess, identifier='tiny', device='cpu')


def _images(tmp_path, n=2):
    paths = []
    for i in range(n):
        path = tmp_path / f'{i}.png'
        Image.fromarray(np.full((8, 8, 3), i * 40, dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


def test_layer_names_are_the_wrapped_module_paths():
    assert _wrapper().layer_names() == ['logits', '0', '1']
