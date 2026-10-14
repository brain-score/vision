"""Cache-key revisions and preprocessing fingerprint for stored temporal activations.

The temporal extractor keys its stored activations on `identifier` and
`stimuli_identifier` only, both raw. That makes it blind to the two things the
image path already accounts for -- a revised plugin and a revised stimulus set --
and to a third, the resolution the model is actually fed.

Fixture mechanics are deliberately the same as
`tests/test_model_helpers/activations/test_cache_key.py`, and several tests here
are the temporal counterpart of one there.
"""
import fnmatch
import inspect
import os
import unittest.mock as mock

import numpy as np
import pytest
from result_caching import get_function_identifier

# No `torch` here, deliberately. `temporal.inputs` pulls in `decord`, after
# which importing torch raises a c10.dll initialisation error on Windows; the
# sibling modules in this directory import it during collection, so a
# module-level `import torch` here would break collection for the whole
# directory. The constraint predates this change -- the unmodified extractor
# already imports `..inputs`. numpy covers the shape assertions, and the one
# test that is about torch's dtype spelling uses a stub, since the probe only
# ever reads `str(dtype)`.

from brainscore_vision.model_helpers.activations import cache_key
from brainscore_vision.model_helpers.activations.cache_key import model_cache_identifier
from brainscore_vision.model_helpers.activations.core import NO_RESOLUTION
from brainscore_vision.model_helpers.activations.temporal.core.extractor import (
    _PROBE_DURATION, _PROBE_FPS, ActivationsExtractor, probe_preprocessing_resolution)
from brainscore_vision.model_helpers.activations.temporal.core.inferencer.base import Inferencer
from brainscore_vision.model_helpers.activations.temporal.inputs import Image, Video


# A registered temporal plugin identifier and the inferencer suffix the
# extractor appends to it. `self.identifier` is the two joined by `@`.
# Derived from the probe's own constants so these expectations cannot drift
# apart from it: the frame count appears in the fingerprint.
PROBE_FRAMES = int(_PROBE_DURATION * _PROBE_FPS / 1000)
CROP112 = f'3x{PROBE_FRAMES}x112x112-float32'
CROP56 = f'3x{PROBE_FRAMES}x56x56-float32'

PLUGIN_IDENTIFIER = 'AVID-CMA-Kinetics400'
INFERENCER_IDENTIFIER = 'TemporalInferencer.dtype=float16.vdeg=8.0'
COMPOSITE_IDENTIFIER = f'{PLUGIN_IDENTIFIER}@{INFERENCER_IDENTIFIER}'


@pytest.fixture(autouse=True)
def _clear_revision_cache():
    """_plugin_revision is lru_cached per process; isolate the tests."""
    cache_key._plugin_revision.cache_clear()
    yield
    cache_key._plugin_revision.cache_clear()


@pytest.fixture
def revisioning_on(monkeypatch):
    """Revisioning is opt-in; most tests here are about what it does when on."""
    monkeypatch.setenv('BRAINSCORE_CACHE_PLUGIN_REVISION', '1')


def stub_extractor(identifier=PLUGIN_IDENTIFIER,
                   inferencer_identifier=INFERENCER_IDENTIFIER,
                   resolution=CROP112):
    """An extractor built the way the image-path tests build theirs.

    Via `__new__`, so no model is loaded and no preprocessing is probed: the
    point of these tests is the call path into the stored call.
    """
    extractor = ActivationsExtractor.__new__(ActivationsExtractor)
    extractor._identifier = identifier
    extractor.inferencer = mock.Mock()
    extractor.inferencer.identifier = inferencer_identifier
    extractor._logger = mock.Mock()
    extractor._resolution = resolution
    extractor._from_paths_stored = mock.Mock(return_value='ASSEMBLY')
    extractor._from_paths = mock.Mock(return_value='ASSEMBLY')
    extractor._reduce_paths = lambda paths: paths
    extractor._expand_paths = lambda activations, original_paths: activations
    return extractor


def stored_kwargs(extractor, stimuli_identifier='hvm-public'):
    """Run `from_paths` and return the kwargs it handed the stored call."""
    result = extractor.from_paths(stimuli_paths=['clip.mp4'], layers=['layer'],
                                  stimuli_identifier=stimuli_identifier)
    assert result == 'ASSEMBLY', "the returned assembly must be untouched"
    return extractor._from_paths_stored.call_args[1]


def build_inferencer(preprocessing, stimulus_type=Video):
    """A real inferencer, which is all the extractor needs to probe."""
    return Inferencer(get_activations=lambda inputs, layers: None,
                      preprocessing=preprocessing,
                      layer_activation_format={'layer': 'CTHW'},
                      stimulus_type=stimulus_type)


def centercrop_preprocessing(crop):
    """A plausible temporal preprocessing: Video -> (C, T, H, W) float32.

    Modelled on `temporal_model_torchvision`'s `get_transform_video`, which is
    the shape the probe has to cope with in production -- in numpy rather than
    torch, for the reason given at the imports.
    """
    def transform_video(video):
        frames = video.to_numpy().astype(np.float32) / 255.  # (T, H, W, C)
        _, height, width, _ = frames.shape
        top, left = (height - crop) // 2, (width - crop) // 2
        frames = frames[:, top:top + crop, left:left + crop, :]
        return frames.transpose(3, 0, 1, 2)  # (C, T, H, W)

    return transform_video


def unwrap_stored():
    """`store_xarray` wraps without `functools.wraps`, so the decorated method
    reports `(*args, **kwargs)`. The undecorated function and the storage that
    holds `identifier_ignore` are reachable through the closure.

    Same helper as in
    `tests/test_model_helpers/activations/test_preprocessing_resolution.py`.
    """
    cells = [cell.cell_contents
             for cell in ActivationsExtractor._from_paths_stored.__closure__]
    function = next(c for c in cells if inspect.isfunction(c))
    storage = next(c for c in cells if hasattr(c, 'identifier_ignore'))
    return function, storage


class TestPluginRevisionReachesTheTemporalKey:
    """The guard the maintainer asked for: a revised temporal plugin must not
    reuse the previous cache key."""

    def test_two_revisions_produce_different_keys(self, revisioning_on, tmp_path, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_MODEL_PLUGIN_SHA', raising=False)
        (tmp_path / 'model.py').write_text('weights = "v1"')
        with mock.patch.object(cache_key, '_locate_plugin_dir', return_value=tmp_path), \
             mock.patch.object(cache_key, '_git_revision', return_value=None):
            key_v1 = stored_kwargs(stub_extractor())['identifier']
            cache_key._plugin_revision.cache_clear()
            (tmp_path / 'model.py').write_text('weights = "v2"')
            key_v2 = stored_kwargs(stub_extractor())['identifier']
        assert key_v1 != key_v2, "revising a temporal plugin must land under a new cache key"
        assert key_v1.startswith(f'{COMPOSITE_IDENTIFIER}@')
        assert key_v2.startswith(f'{COMPOSITE_IDENTIFIER}@')

    def test_env_var_revision_reaches_the_key(self, revisioning_on, monkeypatch):
        monkeypatch.setenv('BRAINSCORE_MODEL_PLUGIN_SHA', 'a' * 40)
        assert stored_kwargs(stub_extractor())['identifier'] == \
               f'{COMPOSITE_IDENTIFIER}@' + 'a' * 12

    def test_the_inferencer_configuration_stays_in_the_key(self, revisioning_on, monkeypatch):
        """dtype/visual degrees/max_spatial_size change the activations, so the
        suffix has to survive into the key: the revision is appended to the
        full identifier, not to the plugin part it was looked up under."""
        monkeypatch.setenv('BRAINSCORE_MODEL_PLUGIN_SHA', 'a' * 40)
        eight = stored_kwargs(stub_extractor(
            inferencer_identifier='TemporalInferencer.dtype=float16.vdeg=8.0'))['identifier']
        ten = stored_kwargs(stub_extractor(
            inferencer_identifier='TemporalInferencer.dtype=float16.vdeg=10.0'))['identifier']
        assert eight != ten
        assert 'vdeg=8.0' in eight and 'vdeg=10.0' in ten

    def test_lookup_is_scoped_to_the_plugin_part(self, revisioning_on, monkeypatch):
        """The composite identifier is registered nowhere. Resolving it whole
        finds no plugin and refuses to cache; the revision has to be looked up
        under the part before the `@`. Pins why `lookup_identifier` exists."""
        monkeypatch.delenv('BRAINSCORE_MODEL_PLUGIN_SHA', raising=False)
        assert model_cache_identifier(COMPOSITE_IDENTIFIER) is None, \
            "if this resolves, lookup_identifier is no longer needed here"
        cache_key._plugin_revision.cache_clear()
        scoped = model_cache_identifier(COMPOSITE_IDENTIFIER,
                                        lookup_identifier=PLUGIN_IDENTIFIER)
        assert scoped is not None and scoped.startswith(f'{COMPOSITE_IDENTIFIER}@')

    def test_a_real_temporal_plugin_is_cacheable(self, revisioning_on, monkeypatch):
        """End to end against the registry, not a mocked plugin dir."""
        monkeypatch.delenv('BRAINSCORE_MODEL_PLUGIN_SHA', raising=False)
        identifier = stored_kwargs(stub_extractor())['identifier']
        assert identifier is not None and identifier.startswith(f'{COMPOSITE_IDENTIFIER}@')

    def test_unresolvable_revision_bypasses_the_cache_entirely(self, revisioning_on, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_MODEL_PLUGIN_SHA', raising=False)
        monkeypatch.delenv('BRAINSCORE_DATA_PLUGIN_SHA', raising=False)
        extractor = stub_extractor()
        with mock.patch.object(cache_key, '_locate_plugin_dir', return_value=None), \
             mock.patch.object(cache_key, '_locate_plugin_dir_by_stimulus_identifier',
                               return_value=None):
            result = extractor.from_paths(stimuli_paths=['clip.mp4'], layers=['layer'],
                                          stimuli_identifier='hvm-public')
        extractor._from_paths_stored.assert_not_called()
        extractor._from_paths.assert_called_once()
        assert result == 'ASSEMBLY', "refusing to cache must still return the assembly"

    def test_revisioning_off_keeps_todays_key(self, monkeypatch):
        """Developers keep their warm caches: with revisioning unset the key
        must be byte-for-byte what it was before this change."""
        monkeypatch.delenv('BRAINSCORE_CACHE_PLUGIN_REVISION', raising=False)
        kwargs = stored_kwargs(stub_extractor())
        assert kwargs['identifier'] == COMPOSITE_IDENTIFIER
        assert kwargs['stimuli_identifier'] == 'hvm-public'

    def test_identifier_property_is_not_mutated(self, revisioning_on, monkeypatch):
        """Only the cache key carries the revision; `identifier` is used
        elsewhere and must stay clean."""
        monkeypatch.setenv('BRAINSCORE_MODEL_PLUGIN_SHA', 'a' * 40)
        extractor = stub_extractor()
        stored_kwargs(extractor)
        assert extractor.identifier == COMPOSITE_IDENTIFIER

    def test_unstored_path_is_untouched(self, revisioning_on):
        """No stimuli_identifier -> no storing -> cache_key never involved."""
        extractor = stub_extractor()
        result = extractor.from_paths(stimuli_paths=['clip.mp4'], layers=['layer'],
                                      stimuli_identifier=None)
        assert result == 'ASSEMBLY'
        extractor._from_paths_stored.assert_not_called()
        extractor._from_paths.assert_called_once()


class TestStimulusSetRevisionReachesTheTemporalKey:
    def test_two_revisions_produce_different_keys(self, revisioning_on, tmp_path, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_DATA_PLUGIN_SHA', raising=False)
        monkeypatch.setenv('BRAINSCORE_MODEL_PLUGIN_SHA', 'a' * 40)
        (tmp_path / 'data.py').write_text('stimuli = "v1"')
        with mock.patch.object(cache_key, '_locate_plugin_dir', return_value=tmp_path), \
             mock.patch.object(cache_key, '_git_revision', return_value=None):
            key_v1 = stored_kwargs(stub_extractor())['stimuli_identifier']
            cache_key._plugin_revision.cache_clear()
            (tmp_path / 'data.py').write_text('stimuli = "v2"')
            key_v2 = stored_kwargs(stub_extractor())['stimuli_identifier']
        assert key_v1 != key_v2, "a revised stimulus set must land under a new cache key"
        assert key_v1.startswith('hvm-public@') and key_v2.startswith('hvm-public@')

    def test_env_var_revision_reaches_the_key(self, revisioning_on, monkeypatch):
        monkeypatch.setenv('BRAINSCORE_MODEL_PLUGIN_SHA', 'a' * 40)
        monkeypatch.setenv('BRAINSCORE_DATA_PLUGIN_SHA', 'b' * 40)
        assert stored_kwargs(stub_extractor())['stimuli_identifier'] == \
               'hvm-public@' + 'b' * 12


class TestPreprocessingFingerprint:
    """The third component, and the one with no counterpart in the old key."""

    def test_same_preprocessing_gives_the_same_component(self):
        first = probe_preprocessing_resolution(centercrop_preprocessing(112), Video)
        second = probe_preprocessing_resolution(centercrop_preprocessing(112), Video)
        assert first == second == CROP112

    def test_different_resolution_gives_a_different_component(self):
        big = probe_preprocessing_resolution(centercrop_preprocessing(112), Video)
        small = probe_preprocessing_resolution(centercrop_preprocessing(56), Video)
        assert big != small
        assert big == CROP112 and small == CROP56

    def test_different_resolution_reaches_the_actual_call_argument(self, monkeypatch):
        """Not just the probe in isolation: the differing component has to
        arrive at `_from_paths_stored`, which is what builds the key."""
        monkeypatch.delenv('BRAINSCORE_CACHE_PLUGIN_REVISION', raising=False)
        big = stored_kwargs(stub_extractor(
            resolution=probe_preprocessing_resolution(centercrop_preprocessing(112), Video)))
        small = stored_kwargs(stub_extractor(
            resolution=probe_preprocessing_resolution(centercrop_preprocessing(56), Video)))
        assert big['resolution'] == CROP112
        assert small['resolution'] == CROP56
        assert big['resolution'] != small['resolution']
        assert big['identifier'] == small['identifier'], \
            "the identifiers are equal -- the resolution is what has to separate these two"

    def test_resolution_is_not_forwarded_into_the_computation(self, monkeypatch):
        """Like the identifiers, it exists only to key; the assembly must not
        depend on it."""
        monkeypatch.delenv('BRAINSCORE_CACHE_PLUGIN_REVISION', raising=False)
        extractor = stub_extractor()
        extractor._from_paths = mock.Mock(return_value='ASSEMBLY')
        stored_kwargs(extractor)
        assert 'resolution' not in inspect.signature(
            ActivationsExtractor._from_paths).parameters

    @pytest.mark.parametrize('dtype,expected', [
        (np.float32, '3x8x8-float32'),
        (np.float16, '3x8x8-float16'),
        (np.uint8, '3x8x8-uint8'),
    ])
    def test_numpy_dtypes_are_named_plainly(self, dtype, expected):
        probe = probe_preprocessing_resolution(
            lambda stimulus: np.zeros((3, 8, 8), dtype=dtype), Video)
        assert probe == expected

    def test_torch_dtype_is_normalised(self):
        """`torch.float32` and numpy's `float32` must not key differently.

        The probe reads the dtype through `str()`, and a torch dtype
        stringifies as `'torch.float32'`, so a stub that stringifies the same
        way exercises the normalisation exactly. Real torch is avoided here for
        the import reason given at the top of the module.
        """
        class TorchLikeTensor:
            shape = (3, 8, 8)
            dtype = 'torch.float32'

        torch_component = probe_preprocessing_resolution(
            lambda stimulus: TorchLikeTensor(), Video)
        numpy_component = probe_preprocessing_resolution(
            lambda stimulus: np.zeros((3, 8, 8), dtype=np.float32), Video)
        assert torch_component == numpy_component == '3x8x8-float32'
        assert 'torch.' not in torch_component

    def test_a_container_of_tensors_is_unpacked(self):
        """Some preprocessings return a per-item container; an element carries
        the shape already."""
        probe = probe_preprocessing_resolution(
            lambda stimulus: [np.zeros((3, 8, 8), np.float32)], Video)
        assert probe == '3x8x8-float32'

    @pytest.mark.parametrize('preprocessing,reason', [
        (lambda stimulus: (_ for _ in ()).throw(RuntimeError('boom')), 'raises'),
        (lambda stimulus: stimulus, 'identity, so no shape'),
        (lambda stimulus: [], 'empty container'),
        (lambda stimulus: 'a string', 'no shape or dtype'),
    ])
    def test_unmeasurable_preprocessing_degrades_to_the_sentinel(self, preprocessing, reason):
        assert probe_preprocessing_resolution(preprocessing, Video) == NO_RESOLUTION, reason

    def test_probe_failure_does_not_break_instantiation(self):
        """The probe runs for every temporal model in the registry; a
        preprocessing it cannot measure must cost a coarser key, never a model
        that cannot be constructed."""
        def exploding_preprocessing(stimulus):
            raise RuntimeError('boom')

        extractor = ActivationsExtractor(inferencer=build_inferencer(exploding_preprocessing),
                                         identifier=PLUGIN_IDENTIFIER)
        assert extractor._resolution == NO_RESOLUTION

    def test_a_measurable_preprocessing_is_probed_on_instantiation(self):
        extractor = ActivationsExtractor(inferencer=build_inferencer(centercrop_preprocessing(112)),
                                         identifier=PLUGIN_IDENTIFIER)
        assert extractor._resolution == CROP112

    def test_an_image_stimulus_type_is_probed_too(self):
        """`Video` is the default, but the inferencer's type decides."""
        probe = probe_preprocessing_resolution(
            lambda stimulus: np.zeros((3, 16, 16), np.float32), Image)
        assert probe == '3x16x16-float32'

    def test_probe_does_not_decode_a_video(self):
        """The probe is built from a still frame on purpose: `decord` and the
        codecs are avoided, and no VideoReader is constructed."""
        import brainscore_vision.model_helpers.activations.temporal.inputs.video as video_module
        with mock.patch.object(video_module, 'VideoReader',
                               side_effect=AssertionError('probe decoded a video')):
            assert probe_preprocessing_resolution(
                centercrop_preprocessing(112), Video) == CROP112

    def test_missing_probe_falls_back_to_the_sentinel(self, monkeypatch):
        """Extractors built with `__new__` never ran `__init__`, so they carry
        no `_resolution`; `from_paths` must still key."""
        monkeypatch.delenv('BRAINSCORE_CACHE_PLUGIN_REVISION', raising=False)
        extractor = stub_extractor()
        del extractor._resolution
        assert stored_kwargs(extractor)['resolution'] == NO_RESOLUTION

    def test_empty_probe_result_falls_back_to_the_sentinel(self, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_CACHE_PLUGIN_REVISION', raising=False)
        assert stored_kwargs(stub_extractor(resolution=''))['resolution'] == NO_RESOLUTION


class TestKeyShape:
    def test_resolution_sits_behind_stimuli_identifier(self):
        """`score_model` prints a cache-cleanup hint globbing
        `identifier=...,stimuli_identifier=*.pkl`. A component inserted ahead
        of `stimuli_identifier` would silently stop matching it."""
        function, _ = unwrap_stored()
        parameters = list(inspect.signature(function).parameters)
        assert parameters.index('resolution') == parameters.index('stimuli_identifier') + 1

    def test_resolution_is_part_of_the_key(self):
        """`identifier_ignore` is what keeps a parameter out of the key."""
        _, storage = unwrap_stored()
        assert 'resolution' not in storage.identifier_ignore, \
            "the fingerprint would not reach the key"
        assert 'stimuli_paths' in storage.identifier_ignore
        assert 'layers' in storage.identifier_ignore

    def test_cleanup_hint_glob_still_matches_the_stored_filename(self):
        """The consequence of that position, measured rather than asserted:
        build the name `result_caching` would store under and match it against
        the glob `brainscore_vision.score_model` prints (`__init__.py:110`).
        The temporal identifier carries the inferencer configuration after an
        `@`, so this also pins that the `@` does not break the hint."""
        function, _ = unwrap_stored()
        key_arguments = dict(identifier=COMPOSITE_IDENTIFIER,
                             stimuli_identifier='hvm-public', resolution=CROP112)
        filename = os.path.basename(get_function_identifier(function, key_arguments)) + '.pkl'
        assert fnmatch.fnmatch(
            filename, f'identifier={COMPOSITE_IDENTIFIER},stimuli_identifier=*.pkl')


class TestPublicSignaturesUnchanged:
    """These are called by wrappers and by `insert_attrs`; this change must not
    alter how any of them is invoked."""

    @pytest.mark.parametrize('method,expected', [
        ('from_paths', ['self', 'stimuli_paths', 'layers', 'stimuli_identifier']),
        ('from_stimulus_set', ['self', 'stimulus_set', 'layers', 'stimuli_identifier']),
        ('__call__', ['self', 'stimuli', 'layers', 'stimuli_identifier',
                      'number_of_trials', 'require_variance']),
    ])
    def test_signature_is_unchanged(self, method, expected):
        parameters = list(inspect.signature(getattr(ActivationsExtractor, method)).parameters)
        assert parameters == expected
