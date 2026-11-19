"""The temporal extractor's store finds a stored result by the bytes of its inputs and the model state.

The table name and the model identifier stay in the key, as in the ordinary helper.
A new file at an old path, or new weights under the same identifier, must give the
values of a cold run. Each case first sees the store write an entry and serve it,
so the test fails when the store is off. The store is a folder of this test alone.
"""
import numpy as np
import pytest
import result_caching
import torch
from PIL import Image

from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper

STORED = 'brainscore_vision.model_helpers.activations.temporal.core.extractor.'


class _Passes:
    """Counts forward passes; a hook with a fixed ``cache_config``, so it does not change the model key."""

    def __init__(self):
        self.count = 0

    def __call__(self, module, args):
        self.count += 1

    def cache_config(self):
        return 'forward pass counter'


@pytest.fixture
def store(monkeypatch, tmp_path):
    state = {'on': True}
    monkeypatch.setattr(result_caching, 'is_enabled', lambda identifier: state['on'] and identifier.startswith(STORED))
    monkeypatch.setattr('brainscore_core.extraction_cache.is_enabled',
                        lambda identifier: state['on'] and identifier.startswith(STORED))
    return state


def _wrapper(tmp_path, monkeypatch):
    with torch.random.fork_rng():
        torch.manual_seed(5)
        network = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    passes = _Passes()
    network.register_forward_pre_hook(passes)

    def frames(video):
        return torch.from_numpy(video.to_numpy()[:, :8, :8].astype(np.float32) / 255)
    wrapper = PytorchWrapper('store-temporal', network, frames, fps=2, img_duration=1000,
                             layer_activation_format={'2': 'THWC'}, batch_size=1, max_workers=1)
    storage = next(cell.cell_contents for cell in wrapper._extractor._from_paths_stored.__func__.__closure__
                   if isinstance(cell.cell_contents, result_caching._Storage))
    monkeypatch.setattr(storage, '_storage_directory', str(tmp_path / 'store'))
    return wrapper, network, passes


def _table(tmp_path):
    rng = np.random.default_rng(3)
    paths = []
    for index in range(2):
        paths.append(str(tmp_path / f'picture{index}.png'))
        Image.fromarray(rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)).save(paths[-1])
    table = StimulusSet({'stimulus_id': ['p0', 'p1'], 'image_path': paths})
    table.stimulus_paths = dict(zip(table['stimulus_id'], paths))
    table.identifier = 'store-temporal-pictures'
    return table


def _new_bytes(table, network):
    Image.new('RGB', (12, 12), (255, 255, 255)).save(table.stimulus_paths['p0'])


def _new_weights(table, network):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(0.25)


@pytest.mark.parametrize('change', [_new_bytes, _new_weights], ids=['new_bytes_at_an_old_path', 'new_weights'])
def test_a_change_under_the_old_names_gives_the_cold_values(change, store, tmp_path, monkeypatch):
    wrapper, network, passes = _wrapper(tmp_path, monkeypatch)
    table = _table(tmp_path)
    first = wrapper(table, layers=['2'])
    assert list((tmp_path / 'store').rglob('*.pkl')), 'the store wrote no entry'
    before = passes.count
    np.testing.assert_array_equal(wrapper(table, layers=['2']).values, first.values)
    assert passes.count == before, 'the store did not serve the stored table'
    change(table, network)
    warm = wrapper(table, layers=['2']).values
    store['on'] = False
    cold = wrapper(table, layers=['2']).values
    assert warm.tobytes() == cold.tobytes()
    assert warm.tobytes() != first.values.tobytes()
