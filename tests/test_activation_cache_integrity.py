"""Use real on-disk cache entries, including changed bytes at the same path."""
import os
from contextlib import nullcontext

import numpy as np
from PIL import Image
import pytest
import torch

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper


@pytest.fixture(params=[False, True], ids=['direct', 'scoring-scope'])
def weight_scope(request):
    from brainscore_core.extraction_cache import weight_fingerprint_scope
    with weight_fingerprint_scope() if request.param else nullcontext():
        yield


def preprocess(paths):
    return np.array([np.asarray(Image.open(path), dtype=np.float32).reshape(-1)
                     for path in paths])


def make_wrapper(weight):
    model = torch.nn.Sequential(torch.nn.Linear(12, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(weight)
    wrapper = PytorchWrapper(model, preprocess, identifier='same-name', batch_size=1)
    return wrapper


@pytest.mark.parametrize('channel', [None, 'vision'])
@pytest.mark.parametrize('change', ['new_model', 'in_place', 'data_assignment', 'image'])
def test_changed_contents_invalidate_real_cache(change, channel, tmp_path, monkeypatch, weight_scope):
    import result_caching
    cache = tmp_path / 'cache'
    # Decorators capture their directories at import time. Redirect the actual
    # storage objects, including xarray backends with their own storage_path().
    from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper
    for name in ('_from_paths_stored', '_from_paths_stored_by_channel'):
        method = getattr(ActivationsExtractorHelper, name)
        storage = next(cell.cell_contents for cell in method.__closure__
                       if isinstance(cell.cell_contents, result_caching._XarrayStorage))
        monkeypatch.setattr(storage, '_storage_directory', str(cache))
        assert os.path.commonpath([str(cache), storage.storage_path('probe')]) == str(cache)

    # Only the test model's activation caches may run; every unrelated cache
    # stays disabled, even if another dependency performs a cached lookup.
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    def enabled(identifier):
        return ('ActivationsExtractorHelper._from_paths_stored' in identifier
                and 'identifier=same-name' in identifier)
    monkeypatch.setattr(result_caching, 'is_enabled', enabled)
    monkeypatch.setattr('brainscore_core.extraction_cache.is_enabled', enabled)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    path = tmp_path / 'image.bmp'
    Image.new('RGB', (2, 2), (1, 1, 1)).save(path)
    wrapper = make_wrapper(1)
    wrapper._extractor.channel = channel

    def extract():
        return wrapper([str(path)], layers=['0'], stimuli_identifier='same-input').values

    before = extract()
    # A real hit must avoid the forward pass, not merely return equal values.
    with monkeypatch.context() as patch:
        patch.setattr(wrapper._model, 'forward', lambda *a, **kw: pytest.fail('cache miss'))
        np.testing.assert_array_equal(extract(), before)
    if change == 'new_model':
        wrapper = make_wrapper(2)
        wrapper._extractor.channel = channel
    elif change == 'in_place':
        with torch.no_grad():
            wrapper._model[0].weight.mul_(2)
    elif change == 'data_assignment':
        wrapper._model[0].weight.data = wrapper._model[0].weight.data * 2
    else:
        stat = path.stat()
        Image.new('RGB', (2, 2), (2, 2, 2)).save(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert path.stat().st_size == stat.st_size
    after = extract()
    np.testing.assert_array_equal(after, before * 2)
    assert len([path for path in cache.rglob('*') if path.is_file()]) == 2
