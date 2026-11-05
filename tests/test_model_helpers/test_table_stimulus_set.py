"""A stimulus set built from a table of image paths, and the declared layer names."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
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


def test_image_path_column_alone_is_enough(tmp_path):
    paths = _images(tmp_path)
    table = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b'], 'image_path': paths}))
    table.identifier = None
    packaged = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b'], 'image_path': paths}))
    packaged.identifier = None
    packaged.stimulus_paths = dict(zip(['a', 'b'], paths))
    wrapper = _wrapper()
    from_table = wrapper(table, layers=['0'])
    assert not hasattr(table, 'stimulus_paths')
    np.testing.assert_array_equal(from_table.values, wrapper(packaged, layers=['0']).values)
    assert list(from_table['stimulus_id'].values) == ['a', 'b']


def test_set_without_files_names_the_column_to_add():
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a']}))
    with pytest.raises(ValueError, match="'image_path' column"):
        _wrapper()(stimuli, layers=['0'])


def test_layer_names_are_the_wrapped_module_paths():
    assert _wrapper().layer_names() == ['logits', '0', '1']
