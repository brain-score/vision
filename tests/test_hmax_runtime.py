"""HMAX integrity and cleanup checks without remote weights or benchmarks."""
import os
import ssl
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def preserve_https_context(monkeypatch):
    monkeypatch.setattr(ssl, '_create_default_https_context',
                        ssl._create_default_https_context)


def test_import_preserves_verified_https():
    code = """
import ssl
original = ssl._create_default_https_context
import brainscore_vision.models.hmax.model
assert ssl._create_default_https_context is original
context = ssl._create_default_https_context()
assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_corrupt_patch_set_is_rejected_before_model_construction(tmp_path, monkeypatch):
    from brainscore_vision.models.hmax import model
    weights = tmp_path / 'universal_patch_set.mat'
    weights.write_bytes(b'corrupted cached checkpoint')
    monkeypatch.setattr(model, 'load_weight_file', lambda **kwargs: weights)
    monkeypatch.setattr(model, 'HMAX', lambda path: pytest.fail('constructed with corrupt weights'))
    with pytest.raises(ValueError, match='SHA1 mismatch'):
        model.get_model('hmax')


@pytest.mark.parametrize('previous', [None, '', '1', 'another.module'])
@pytest.mark.parametrize('fail', [False, True])
def test_activation_cache_controls_are_preserved_and_restored(monkeypatch, previous, fail):
    from result_caching import is_enabled
    from brainscore_vision.models.hmax.helpers.pytorch import PytorchWrapper
    if previous is None:
        monkeypatch.delenv('RESULTCACHING_DISABLE', raising=False)
    else:
        monkeypatch.setenv('RESULTCACHING_DISABLE', previous)
    observed = []

    def extract(*args, **kwargs):
        observed.append({name: is_enabled(name) for name in (
            'brainscore_vision.model_helpers.activations.extract',
            'model_tools.activations.extract', 'another.module.function',
            'unrelated.module.function')})
        if fail:
            raise RuntimeError('extraction failed')
        return 'activations'

    wrapper = PytorchWrapper.__new__(PytorchWrapper)
    wrapper._extractor = extract
    if fail:
        with pytest.raises(RuntimeError, match='extraction failed'):
            wrapper('images')
    else:
        assert wrapper('images') == 'activations'
    restored = os.environ.get('RESULTCACHING_DISABLE')
    assert restored == previous
    assert not observed[0]['brainscore_vision.model_helpers.activations.extract']
    assert not observed[0]['model_tools.activations.extract']
    if previous in ('1', 'another.module'):
        assert not observed[0]['another.module.function']
    assert observed[0]['unrelated.module.function'] is (previous != '1')


@pytest.mark.parametrize('failure', [None, 'lookup', 'forward'])
def test_temporary_hooks_are_removed_without_removing_existing_hooks(failure):
    import torch
    from brainscore_vision.models.hmax.helpers.pytorch import PytorchWrapper

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Identity()

        def forward(self, value):
            value = self.first(value)
            if failure == 'forward':
                raise RuntimeError('forward failed')
            return value

    model = Model()
    existing = model.first.register_forward_hook(lambda *args: None)
    original_hooks = dict(model.first._forward_hooks)
    wrapper = PytorchWrapper.__new__(PytorchWrapper)
    wrapper._model, wrapper._device = model, torch.device('cpu')
    images = [np.ones(2, dtype=np.float32)]
    try:
        if failure == 'lookup':
            with pytest.raises(AssertionError, match='No submodule'):
                wrapper.get_activations(images, ['first', 'missing'])
        elif failure == 'forward':
            with pytest.raises(RuntimeError, match='forward failed'):
                wrapper.get_activations(images, ['first'])
        else:
            result = wrapper.get_activations(images, ['first'])
            np.testing.assert_array_equal(result['first'], np.ones((1, 2)))
        assert dict(model.first._forward_hooks) == original_hooks
    finally:
        existing.remove()
