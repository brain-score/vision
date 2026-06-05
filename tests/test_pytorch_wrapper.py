"""Unit tests for PytorchWrapper — the foundational vision activations wrapper.

The coverage audit flagged 0 direct tests for
`brainscore_vision/model_helpers/activations/pytorch.py`, yet every vision model
routes through it. These exercise its core mechanics on a tiny CPU model (no
weights, no S3, no caching): hook capture, dotted-path layer resolution, the
'logits' output layer, tuple-output handling, batching, dtype matching, the
input_key/forward_kwargs paths, and leaf-layer enumeration.
"""
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper


class _Tiny(nn.Module):
    """3->4 conv (nested in a Sequential) -> pool -> 4->2 linear."""
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 4, 3, padding=1)), ('relu', nn.ReLU())]))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _TupleOut(nn.Module):
    """A block whose forward returns a tuple (like many transformer layers)."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x), None


class _KwargModel(nn.Module):
    """forward takes a keyword argument (exercises input_key)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, pixel_values=None):
        return self.fc(pixel_values.mean(dim=(2, 3)))


def _wrap(model, **kw):
    return PytorchWrapper(model=model, preprocessing=lambda paths: paths,
                          identifier='tiny-test', **kw)


@pytest.mark.unit
class TestHookCapture:
    def test_captures_named_layers(self):
        w = _wrap(_Tiny())
        img = np.random.RandomState(0).rand(3, 8, 8).astype(np.float32)
        acts = w.get_activations([img], ['block.conv', 'fc'])
        assert set(acts) == {'block.conv', 'fc'}
        assert acts['block.conv'].shape == (1, 4, 8, 8)
        assert acts['fc'].shape == (1, 2)

    def test_batching(self):
        w = _wrap(_Tiny())
        imgs = [np.random.RandomState(i).rand(3, 8, 8).astype(np.float32) for i in range(3)]
        acts = w.get_activations(imgs, ['fc'])
        assert acts['fc'].shape == (3, 2)

    def test_hooks_removed_after_call(self):
        m = _Tiny(); w = _wrap(m)
        img = np.random.rand(3, 8, 8).astype(np.float32)
        w.get_activations([img], ['fc'])
        # no lingering forward hooks on the layer
        assert len(w.get_layer('fc')._forward_hooks) == 0


@pytest.mark.unit
class TestLayerResolution:
    def test_dotted_path(self):
        w = _wrap(_Tiny())
        assert isinstance(w.get_layer('block.conv'), nn.Conv2d)

    def test_logits_returns_output_layer(self):
        w = _wrap(_Tiny())
        assert w.get_layer('logits') is w._model.fc

    def test_invalid_layer_raises(self):
        w = _wrap(_Tiny())
        with pytest.raises(AssertionError):
            w.get_layer('does.not.exist')

    def test_layers_yields_only_leaves(self):
        w = _wrap(_Tiny())
        names = {n for n, _ in w.layers()}
        assert 'block.conv' in names and 'fc' in names
        assert 'block' not in names          # container, not a leaf


@pytest.mark.unit
class TestOutputHandling:
    def test_tuple_output_takes_first(self):
        w = _wrap(_TupleOut())
        x = np.random.rand(2, 4).astype(np.float32)
        acts = w.get_activations([x[i] for i in range(2)], ['lin'])
        assert acts['lin'].shape == (2, 4)        # unwrapped from the tuple

    def test_tensor_to_numpy_unwraps_tuple(self):
        t = torch.arange(6.).reshape(2, 3)
        assert np.allclose(PytorchWrapper._tensor_to_numpy((t, None)), t.numpy())


@pytest.mark.unit
class TestInputPaths:
    def test_input_key(self):
        w = _wrap(_KwargModel(), input_key='pixel_values')
        img = np.random.rand(3, 8, 8).astype(np.float32)
        acts = w.get_activations([img], ['fc'])
        assert acts['fc'].shape == (1, 2)

    def test_dtype_matches_model(self):
        # fp16 model + fp32 input -> wrapper casts input to model dtype, no error
        m = _Tiny().half()
        w = _wrap(m)
        img = np.random.rand(3, 8, 8).astype(np.float32)
        acts = w.get_activations([img], ['fc'])
        assert acts['fc'].shape == (1, 2)
