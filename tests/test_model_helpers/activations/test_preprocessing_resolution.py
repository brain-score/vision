"""The preprocessing resolution as part of the stored-activations cache key.

Two properties are under test. First, the component must actually distinguish
what it claims to: two commitments sharing an identifier but preprocessing to
different resolutions must not share a cache entry. Second, measuring it must
never be able to fail a scoring run -- the probe runs while an extractor is
being constructed, so anything it cannot measure degrades to `NO_RESOLUTION`
rather than raising.

These use stub preprocessing callables and a stub extractor rather than real
models: the point is the call path and the key, and loading models here would
make the suite unrunnable locally.
"""
import fnmatch
import functools
import inspect
import os
import unittest.mock as mock

import numpy as np
import pytest
import torch

from brainscore_vision.model_helpers.activations.core import (
    ActivationsExtractorHelper, NO_RESOLUTION, _probe_image, probe_preprocessing_resolution)
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from result_caching import get_function_identifier


def preprocessing_at(image_size):
    """The idiom 519 of the 552 image plugins use, at a chosen resolution."""
    return functools.partial(load_preprocess_images, image_size=image_size)


def build_extractor(preprocessing, identifier='alexnet'):
    """A real extractor -- `__init__` runs, so the probe runs."""
    return ActivationsExtractorHelper(get_activations=mock.Mock(), preprocessing=preprocessing,
                                      identifier=identifier)


def unwrap_stored():
    """`store_xarray` wraps without `functools.wraps`, so the decorated method
    reports `(*args, **kwargs)`. The undecorated function and the storage that
    holds `identifier_ignore` are reachable through the closure."""
    cells = [cell.cell_contents for cell in ActivationsExtractorHelper._from_paths_stored.__closure__]
    function = next(c for c in cells if inspect.isfunction(c))
    storage = next(c for c in cells if hasattr(c, 'identifier_ignore'))
    return function, storage


class TestComponentDistinguishesResolutions:
    """The reason the component exists: same identifier, different input size."""

    def test_same_resolution_same_component(self):
        """Guards against a probe that is not deterministic -- a component that
        varied per call would key every request separately and disable the cache."""
        assert (probe_preprocessing_resolution(preprocessing_at(224))
                == probe_preprocessing_resolution(preprocessing_at(224)))

    def test_different_resolution_different_component(self):
        assert (probe_preprocessing_resolution(preprocessing_at(224))
                != probe_preprocessing_resolution(preprocessing_at(64)))

    def test_component_reports_shape_and_dtype(self):
        assert probe_preprocessing_resolution(preprocessing_at(224)) == '3x224x224-float32'

    def test_component_is_filename_safe(self):
        """It is spliced into a path and into a `,`-joined parameter list."""
        component = probe_preprocessing_resolution(preprocessing_at(224))
        assert not (set(component) & set('/\\:,; '))


class TestWhatGoesIntoTheComponent:

    def test_batch_dimension_is_not_in_the_component(self):
        """The leading axis is the number of probe images, not a property of the
        model, so probing with one or two images must give the same answer."""
        preprocess, image = preprocessing_at(224), _probe_image()
        one = probe_preprocessing_resolution(preprocess, probe_paths=[image] * 1)
        two = probe_preprocessing_resolution(preprocess, probe_paths=[image] * 2)
        assert one == two == '3x224x224-float32'

    def test_dtype_is_normalised_across_frameworks(self):
        """`str(torch.float32)` is `'torch.float32'`, numpy's is `'float32'`.
        Unnormalised, two plugins at the same resolution would key differently."""
        as_torch = probe_preprocessing_resolution(lambda paths: torch.zeros(len(paths), 3, 224, 224))
        as_numpy = probe_preprocessing_resolution(
            lambda paths: np.zeros((len(paths), 3, 224, 224), dtype=np.float32))
        assert as_torch == as_numpy == '3x224x224-float32'

    def test_list_container_is_unwrapped(self):
        """`omnivore_*` returns a list of per-image tensors rather than a stacked
        array; the resolution is in there and must still be found."""
        preprocess = lambda paths: [torch.zeros(3, 1, 224, 224) for _ in paths]
        assert probe_preprocessing_resolution(preprocess) == '3x1x224x224-float32'

    def test_list_element_shape_is_already_non_batch(self):
        """The list length is the batch, so no axis may be dropped from the element."""
        preprocess = lambda paths: [torch.zeros(3, 224, 224) for _ in paths]
        assert probe_preprocessing_resolution(preprocess) == '3x224x224-float32'


class TestProbeFailureFallsBackToTheSentinel:
    """`NO_RESOLUTION` rather than an exception or a silently absent component:
    the key stays as good as it was before this existed, and the filename shows
    where the coverage ends.
    """

    def test_identity_preprocessing_is_unprobed(self):
        """`preprocessing=None` becomes `lambda x: x` in `__init__`, so the probe
        gets its own paths back. Five plugins do this (`pixels`, `gabor_filter_*`);
        they apply their resolution inside `get_activations` instead."""
        extractor = build_extractor(preprocessing=None)
        assert extractor._resolution == NO_RESOLUTION

    def test_raising_preprocessing_is_unprobed(self):
        def preprocess(paths):
            raise RuntimeError("this plugin cannot be probed")
        assert probe_preprocessing_resolution(preprocess) == NO_RESOLUTION

    def test_probe_never_raises_out_of_init(self):
        """The constructor runs for every model in the registry; a preprocessing
        that cannot be probed must not be a model that cannot be built."""
        def preprocess(paths):
            raise RuntimeError("this plugin cannot be probed")
        extractor = build_extractor(preprocessing=preprocess)
        assert extractor._resolution == NO_RESOLUTION

    @pytest.mark.parametrize('preprocess,reason', [
        (lambda paths: paths, "identity, elements are str"),
        (lambda paths: [], "empty batch"),
        (lambda paths: np.zeros(4), "only a batch axis, no resolution left"),
        (lambda paths: 'not an array', "no shape at all"),
    ])
    def test_unmeasurable_results_are_unprobed(self, preprocess, reason):
        assert probe_preprocessing_resolution(preprocess) == NO_RESOLUTION, reason


class TestComponentReachesTheCacheKey:
    """A component that never arrives at `_from_paths_stored` keys nothing."""

    def _extractor_with_mocked_store(self, preprocessing):
        extractor = build_extractor(preprocessing)
        extractor._from_paths_stored = mock.Mock(return_value='ASSEMBLY')
        extractor._from_paths = mock.Mock(return_value='ASSEMBLY')
        return extractor

    def _resolution_passed_by(self, preprocessing):
        extractor = self._extractor_with_mocked_store(preprocessing)
        result = extractor.from_paths(stimuli_paths=['x.png'], layers=['fc'],
                                      stimuli_identifier='Papale2025')
        assert result == 'ASSEMBLY', "the returned assembly must be untouched"
        return extractor._from_paths_stored.call_args[1]['resolution']

    def test_different_resolutions_reach_the_stored_call_differently(self):
        """The collision from the issue, at the boundary that actually keys."""
        assert self._resolution_passed_by(preprocessing_at(224)) == '3x224x224-float32'
        assert self._resolution_passed_by(preprocessing_at(64)) == '3x64x64-float32'

    def test_sentinel_reaches_the_stored_call(self):
        assert self._resolution_passed_by(None) == NO_RESOLUTION

    def test_resolution_is_not_ignored_when_building_the_key(self):
        """`identifier_ignore` decides what `store_xarray` leaves out of the key.
        Listing `resolution` there would make the whole change a no-op."""
        _, storage = unwrap_stored()
        assert 'resolution' not in storage.identifier_ignore

    def test_extractor_built_without_init_still_keys(self):
        """`test_cache_key.py` stubs extractors with `__new__`, so `__init__` never
        runs and there is no probed attribute. Reading it must not raise."""
        extractor = ActivationsExtractorHelper.__new__(ActivationsExtractorHelper)
        extractor.identifier = 'alexnet'
        extractor._logger = mock.Mock()
        extractor._from_paths_stored = mock.Mock(return_value='ASSEMBLY')
        extractor._from_paths = mock.Mock(return_value='ASSEMBLY')
        extractor._reduce_paths = lambda paths: paths
        extractor._expand_paths = lambda a, original_paths: a
        extractor.from_paths(stimuli_paths=['x.png'], layers=['fc'],
                             stimuli_identifier='Papale2025')
        assert extractor._from_paths_stored.call_args[1]['resolution'] == NO_RESOLUTION


class TestPublicSignaturesUnchanged:
    """Only the private `_from_paths_stored` gains a parameter. Plugins call the
    public methods -- five of them via `super().from_paths(*args, **kwargs)`.
    """

    @pytest.mark.parametrize('method,parameters', [
        ('from_paths', ['self', 'stimuli_paths', 'layers', 'stimuli_identifier', 'require_variance']),
        ('from_stimulus_set', ['self', 'stimulus_set', 'layers', 'stimuli_identifier', 'require_variance']),
        ('__call__', ['self', 'stimuli', 'layers', 'stimuli_identifier', 'number_of_trials',
                      'require_variance']),
    ])
    def test_public_signature_is_untouched(self, method, parameters):
        signature = inspect.signature(getattr(ActivationsExtractorHelper, method))
        assert list(signature.parameters) == parameters

    def test_resolution_sits_behind_stimuli_identifier(self):
        """Position is load-bearing: `brainscore_vision.score_model` prints a
        cache-cleanup hint globbing `identifier=...,stimuli_identifier=*.pkl`,
        and `result_caching` orders key parameters by signature position. A
        component ahead of `stimuli_identifier` makes that hint match nothing.
        """
        function, _ = unwrap_stored()
        parameters = list(inspect.signature(function).parameters)
        assert parameters.index('resolution') == parameters.index('stimuli_identifier') + 1

    def test_cleanup_hint_glob_still_matches_the_stored_filename(self):
        """The consequence of that position, measured rather than asserted: build
        the name `result_caching` would store under and match it against the glob
        `brainscore_vision.score_model` prints. Kept as a literal here because it
        is built inline in an f-string there and cannot be imported."""
        function, _ = unwrap_stored()
        key_arguments = dict(identifier='alexnet', stimuli_identifier='hvm-public',
                             resolution='3x224x224-float32', number_of_trials=1,
                             require_variance=False)
        filename = os.path.basename(get_function_identifier(function, key_arguments)) + '.pkl'
        assert fnmatch.fnmatch(filename, 'identifier=alexnet,stimuli_identifier=*.pkl')
