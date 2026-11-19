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


# -- everything that decides a stored value is in the key, or the store is off ----

def _build(tmp_path, monkeypatch, **changes):
    """``_wrapper`` with ``changes`` to its constructor arguments (the wrapper's and the inferencer's)."""
    with torch.random.fork_rng():
        torch.manual_seed(5)
        network = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    passes = _Passes()
    network.register_forward_pre_hook(passes)
    arguments = {**_arguments(), **changes}
    wrapper = PytorchWrapper(arguments.pop('identifier'), network, arguments.pop('preprocessing'), **arguments)
    storage = next(cell.cell_contents for cell in wrapper._extractor._from_paths_stored.__func__.__closure__
                   if isinstance(cell.cell_contents, result_caching._Storage))
    monkeypatch.setattr(storage, '_storage_directory', str(tmp_path / 'store'))
    return wrapper, network, passes


def _frames(video):
    return torch.from_numpy(video.to_numpy()[:, :8, :8].astype(np.float32) / 255)


def _arguments():
    from brainscore_vision.model_helpers.activations.temporal.model.pytorch import default_process_activation
    return dict(identifier='store-temporal', preprocessing=_frames, process_output=default_process_activation, fps=2, img_duration=1000,
                layer_activation_format={'2': 'THWC'}, batch_size=1, max_workers=1)


def _stored(wrapper, passes, table, tmp_path):
    first = wrapper(table, layers=['2'])
    assert list((tmp_path / 'store').rglob('*.pkl')), 'the store wrote no entry'
    before = passes.count
    np.testing.assert_array_equal(wrapper(table, layers=['2']).values, first.values)
    assert passes.count == before, 'the store did not serve the stored table'
    return first


def _served(wrapper, passes, table):
    """True when the store gave ``table`` with no forward pass; an extraction that fails did compute."""
    before = passes.count
    try:
        wrapper(table, layers=['2'])
    except Exception:                                            # noqa: BLE001 - it ran, with a value it refuses
        return False
    return passes.count == before


def test_another_output_callback_gives_the_cold_values(store, tmp_path, monkeypatch):
    wrapper, network, passes = _build(tmp_path, monkeypatch)
    table = _table(tmp_path)
    first = _stored(wrapper, passes, table, tmp_path)
    zeros, _, zero_passes = _build(tmp_path, monkeypatch,
                                   process_output=lambda layer, name, inputs, output: torch.zeros_like(output))
    warm = zeros(table, layers=['2']).values
    store['on'] = False
    cold = zeros(table, layers=['2']).values
    assert not np.any(cold) and np.any(first.values)
    assert warm.tobytes() == cold.tobytes()


def test_a_named_table_with_the_store_off_runs_in_the_order_of_the_store(store, tmp_path, monkeypatch):
    """The stored body runs the files in sorted order; with the store off a named table runs in that order too."""
    def first_of_batch(layer, name, inputs, output):
        return output[:1].expand_as(output)
    wrapper, _, _ = _build(tmp_path, monkeypatch, process_output=first_of_batch, batch_size=2)
    paths = []
    for index in range(16):
        paths.append(str(tmp_path / f'order{index:02}.png'))
        Image.new('RGB', (12, 12), (index * 9, index, 255 - index)).save(paths[-1])
    pair = next([a, b] for a in paths for b in paths if a != b and list({a, b}) != sorted({a, b}))
    stored = wrapper(pair, layers=['2'], stimuli_identifier='named-order').values
    store['on'] = False
    assert wrapper(pair, layers=['2'], stimuli_identifier='named-order').values.tobytes() == stored.tobytes()


def test_the_file_name_of_an_entry_fits_the_limit_of_one_path_component(store, tmp_path, monkeypatch):
    long_name = 'algonauts2025-friends-sub01-frames-' + 'x' * 220
    wrapper, _, _ = _build(tmp_path, monkeypatch, identifier='m' * 200)
    table = _table(tmp_path)
    table.identifier = long_name
    wrapper(table, layers=['2'])
    entries = list((tmp_path / 'store').rglob('*.pkl'))
    assert entries and all(len(part.encode()) + len('.filepart') <= 255
                           for entry in entries for part in entry.relative_to(tmp_path).parts)


def _constructor_arguments():
    """Every argument of the temporal wrapper and of its default inferencer, from their signatures."""
    import inspect
    from brainscore_vision.model_helpers.activations.temporal.model.base import ActivationWrapper
    inferencer = inspect.signature(ActivationWrapper).parameters['inferencer_cls'].default
    names = set()
    for cls in (PytorchWrapper, ActivationWrapper, *inferencer.__mro__):
        if '__init__' in vars(cls):
            names |= {name for name, parameter in inspect.signature(vars(cls)['__init__']).parameters.items()
                      if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)}
    return sorted(names - {'self', 'get_activations', 'stimulus_type', 'model', 'inferencer_cls'})


def _others(value, owners):
    """Other values of the type of ``value``: a number grows, a flag turns, a function gets other code with
    the same result, a string becomes another name of a module the owners' classes use (where ``value`` is
    such a name). Anything else gives none."""
    import inspect
    import types
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1]
    if isinstance(value, float):
        return [value * 2]
    if isinstance(value, str):
        modules = {module for owner in owners for cls in type(owner).__mro__
                   for module in vars(inspect.getmodule(cls) or types).values() if isinstance(module, types.ModuleType)
                   and inspect.isfunction(getattr(module, value, None))}
        return sorted({name for module in modules for name, member in vars(module).items()
                       if inspect.isfunction(member) and member.__module__ == module.__name__ and name != value})
    if callable(value) and getattr(value, '__code__', None) is not None:
        return [lambda *args, _value=value, **kwargs: _value(*args, **kwargs)]
    return []


def _parts(wrapper):
    """The wrapper, its inferencer and the inferencer's executor: the objects that make a stored value."""
    inferencer = wrapper._extractor.inferencer
    return [wrapper, inferencer, vars(inferencer)['_executor']]


def test_every_constructor_argument_is_in_the_key_or_turns_the_store_off(store, tmp_path, monkeypatch):
    """Each argument (from the signatures): another value is another key or no store. The identifier is in
    the name of the entry. An argument the test cannot vary must be kept as ``<name>`` or ``_<name>`` on a part
    (each attribute of a part is in the key: the next test); a new argument is covered so, or named here."""
    wrapper, _, passes = _build(tmp_path, monkeypatch)
    table = _table(tmp_path)
    _stored(wrapper, passes, table, tmp_path)
    given, escaped = _arguments(), []
    defaults = {}
    import inspect
    for cls in type(_parts(wrapper)[1]).__mro__:
        if '__init__' in vars(cls):
            defaults.update({name: p.default for name, p in inspect.signature(vars(cls)['__init__']).parameters.items()
                             if p.default is not p.empty})
    for name in _constructor_arguments():
        if name == 'identifier':
            continue
        value = given.get(name, defaults.get(name))
        others = _others(value, _parts(wrapper))
        if others:
            changed, _, changed_passes = _build(tmp_path, monkeypatch, **{name: others[0]})
            # A hit is right only when the argument left every part as it was (the next test shows that
            # every attribute of a part is in the key); an overridden argument has no effect.
            same_state = _described(changed) == _described(wrapper)
            if _served(changed, changed_passes, table) and not same_state:
                escaped.append(f'{name}={others[0]!r} was served the entry of {name}={value!r}')
        elif not any({name, f'_{name}'} & set(vars(part)) for part in _parts(wrapper)):
            escaped.append(f'{name} cannot be varied here and is kept on no part')
    assert not escaped, f'an argument that can change a stored value is not in the key: {escaped}'


def _described(wrapper):
    from brainscore_core.extraction_cache import fingerprint
    return fingerprint(_parts(wrapper)[1].cache_config())


class _Opaque:
    """An object with no description: a key cannot hold it."""


def test_every_attribute_of_a_part_is_in_the_key_or_turns_the_store_off(store, tmp_path, monkeypatch):
    """Each attribute of the wrapper, the inferencer and the executor after one extraction: another value is
    another key or no store. Not settings: the identifier (in the entry name), the model (its own part of the
    key), the device, the links between the parts, and the scratch of one call (cleared before it is read)."""
    wrapper, _, passes = _build(tmp_path, monkeypatch)
    table = _table(tmp_path)
    _stored(wrapper, passes, table, tmp_path)
    links = {'_extractor', '_executor', 'get_activations', 'from_stimulus_set', 'from_paths',
             'register_batch_activations_hook', 'register_stimulus_set_hook'}
    scratch = {'stimuli', '_mapper', '_logger', '_time_ends'}
    escaped = []
    for part in _parts(wrapper):
        for name, value in list(vars(part).items()):
            if name in links | scratch | {'identifier', '_model', '_device'}:
                continue
            other = (_others(value, _parts(wrapper)) or [_Opaque()])[0]
            setattr(part, name, other)
            if _served(wrapper, passes, table):
                escaped.append(f'{type(part).__name__}.{name}: {other!r} was served the entry of {value!r}')
            setattr(part, name, value)
    assert not escaped, f'an attribute is not in the key: {escaped}'


def test_ambient_autocast_is_in_the_key(store, tmp_path, monkeypatch):
    wrapper, _, passes = _build(tmp_path, monkeypatch)
    table = _table(tmp_path)
    _stored(wrapper, passes, table, tmp_path)
    with torch.autocast('cpu', dtype=torch.float16):
        assert not _served(wrapper, passes, table)
