"""A legacy ModelCommitment with a lazy layer search declares the regions it can map.

A ``ModelCommitment`` with no ``region_layer_map/<id>.json`` and no
``region_layer_map=`` argument builds a lazy ``RegionLayerMap``: it stays empty
until a region is first used, and then a ``LayerSelection`` search picks the
layer. The adapter declares ``neural:<region>`` for every region that search can
map, without running it, and the search runs where it always ran: at the first
legacy ``look_at`` of the region.

These tests use the real ``ModelCommitment``, ``RegionLayerMap``,
``LayerMappedModel`` and ``TemporalAligned``. Only the layer search and the
activations are stubs, so no weights or benchmark data are loaded.
"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import brainscore_vision
from brainscore_core.compatibility import (CompatibilityError, check_compatibility,
                                           ensure_legacy_benchmark_modalities)
from brainscore_core.legacy_facade import LegacyBenchmarkFacade, mark_legacy_benchmark
from brainscore_core.memory import check_memory
from brainscore_core.metrics import Score
from brainscore_core.streaming_helpers import neural_response
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision.compat.unified_adapter import VisionModelAdapter
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment, RegionLayerMap

from tests.test_unified_adapter import _make_legacy_model

STANDARD_REGIONS = {'neural:V1', 'neural:V2', 'neural:V4', 'neural:IT'}


def _stimuli(n=2):
    ids = [f's{i}' for i in range(n)]
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ids,
                                        'image_path': [f'{i}.png' for i in ids],
                                        'object_name': ['cat', 'dog'][:n]}))
    stimuli.identifier = 'synthetic-lazy-regions'
    return stimuli


class _Log(list):
    """One ordered record of layer searches and activation extractions."""


class _CountingLayerSelection:
    """Stands in for LayerSelection: one call is one layer search."""

    def __init__(self, log, layer='l2'):
        self.log, self.layer = log, layer

    def __call__(self, selection_identifier, benchmark):
        self.log.append(('search', selection_identifier))
        return self.layer

    @property
    def searches(self):
        return [entry for entry in self.log if entry[0] == 'search']


class _ActivationsModel:
    """Stands in for the activations model: logs the layers it extracts."""
    identifier = 'stub-activations'

    def __init__(self, log):
        self.log = log
        self._extractor = MagicMock()

    def __call__(self, stimuli, layers, number_of_trials=1, require_variance=None):
        self.log.append(('activations', tuple(layers), len(stimuli)))
        n = len(stimuli)
        return NeuroidAssembly(
            np.arange(n * 2 * len(layers), dtype=float).reshape(n, 2 * len(layers)),
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'object_name': ('presentation', list(stimuli['object_name'])),
                    'neuroid_id': ('neuroid', [f'{layer}.{i}' for layer in layers for i in range(2)]),
                    'layer': ('neuroid', [layer for layer in layers for _ in range(2)])},
            dims=['presentation', 'neuroid'])


def _lazy_commitment(log=None):
    """A real ModelCommitment with no committed map, so its map is lazy."""
    log = _Log() if log is None else log
    model = ModelCommitment('lazy-regions-test-no-such-plugin', _ActivationsModel(log), layers=['l1', 'l2'])
    region_layer_map = model.layer_model.region_layer_map
    assert isinstance(region_layer_map, RegionLayerMap) and len(region_layer_map) == 0
    selection = _CountingLayerSelection(log)
    region_layer_map.layer_selection = selection
    return model, selection


class _LegacyNeuralBenchmark:
    """What a legacy NeuralBenchmark exposes to pre-flight and does to its candidate."""
    identifier = 'MajajHong2015public.IT-pls'
    region = 'IT'
    timebins = [(70, 170)]

    def __init__(self):
        self.stimulus_set = _stimuli()

    def preallocate_memory(self, *args, **kwargs):
        pass

    def __call__(self, candidate):
        candidate.start_recording('IT', time_bins=self.timebins)
        candidate.look_at(self.stimulus_set)
        candidate.look_at(self.stimulus_set)  # a second look_at reuses the committed layer
        return Score([1.0], coords={'aggregation': ['center']}, dims=['aggregation'])


def _legacy_benchmark():
    return mark_legacy_benchmark(ensure_legacy_benchmark_modalities(_LegacyNeuralBenchmark(), {'vision'}))


class TestLazyRegionDeclaration:

    def test_lazy_map_declares_every_wired_region_without_a_search(self):
        model, selection = _lazy_commitment()
        adapter = VisionModelAdapter(model)
        assert adapter.out_channels == STANDARD_REGIONS | {'behavior'}
        assert adapter.region_layer_map == {}
        assert selection.searches == []

    def test_declaration_is_the_same_before_and_after_a_search(self):
        model, selection = _lazy_commitment()
        adapter = VisionModelAdapter(model)
        before = adapter.out_channels
        model.start_recording('IT', [(70, 170)])
        model.look_at(_stimuli())
        assert selection.searches == [('search', 'IT')]
        assert adapter.out_channels == before
        assert adapter.region_layer_map == {'IT': 'l2'}

    def test_any_map_with_public_region_benchmarks_declares_its_regions(self):
        """Discovery reads the public declaration, not the RegionLayerMap class."""

        class LazyMap(dict):
            def __init__(self, region_benchmarks, selection):
                super().__init__()
                self.region_benchmarks = region_benchmarks
                self._selection = selection

            def __getitem__(self, region):
                if region not in self:
                    self[region] = self._selection(selection_identifier=region,
                                                   benchmark=self.region_benchmarks[region])
                return dict.__getitem__(self, region)

        log = _Log()
        model, _ = _lazy_commitment(log)
        selection = _CountingLayerSelection(log)
        model.layer_model.region_layer_map = LazyMap({'IT': '<IT bench>'}, selection)
        adapter = VisionModelAdapter(model)
        assert adapter.out_channels == {'neural:IT', 'behavior'}
        assert selection.searches == []

    def test_probe_safe_regions_are_the_committed_regions(self):
        model, selection = _lazy_commitment()
        adapter = VisionModelAdapter(model)
        assert adapter.probe_safe_regions == []
        model.start_recording('IT', [(70, 170)])
        model.look_at(_stimuli())
        assert adapter.probe_safe_regions == ['IT']
        assert selection.searches == [('search', 'IT')]

    def test_legacy_neural_benchmark_is_a_full_match_without_a_search(self):
        model, selection = _lazy_commitment()
        match = check_compatibility(VisionModelAdapter(model), _legacy_benchmark())
        assert match.kind == 'full'
        assert selection.searches == []


class TestSearchRunsAtFirstUse:

    def test_legacy_benchmark_through_the_facade_makes_the_plain_legacy_calls(self):
        plain_log = _Log()
        plain, plain_selection = _lazy_commitment(plain_log)
        _LegacyNeuralBenchmark()(plain)

        log = _Log()
        model, selection = _lazy_commitment(log)
        _LegacyNeuralBenchmark()(LegacyBenchmarkFacade(VisionModelAdapter(model), visual_degrees=8))

        assert selection.searches == plain_selection.searches == [('search', 'IT')]
        assert log == plain_log
        assert log[0] == ('search', 'IT')

    def test_scoring_runs_the_legacy_benchmark_instead_of_n_a(self, monkeypatch):
        log = _Log()
        model, selection = _lazy_commitment(log)
        monkeypatch.setattr(brainscore_vision, 'import_plugin', lambda *a, **k: None)
        monkeypatch.setitem(brainscore_vision.benchmark_registry, 'legacy-lazy-test', _LegacyNeuralBenchmark)
        monkeypatch.setattr(brainscore_vision, 'load_model', lambda _: VisionModelAdapter(model))
        score = brainscore_vision._run_score('m', 'legacy-lazy-test', check_mem=False)
        assert score.attrs['compatibility']['match'] == 'full'
        assert selection.searches == [('search', 'IT')]

    def test_unified_session_requesting_a_region_searches_once(self):
        log = _Log()
        model, selection = _lazy_commitment(log)
        adapter = VisionModelAdapter(model)
        first = neural_response(adapter, _stimuli(), ['neural:IT'])
        second = neural_response(adapter, _stimuli(), ['neural:IT'])
        assert selection.searches == [('search', 'IT')]
        assert set(first['layer'].values) == set(second['layer'].values) == {'l2'}

    def test_memory_probe_runs_the_search_once_and_the_benchmark_reuses_it(self):
        """The one-stimulus probe is the first use of the region, so it runs the search."""
        log = _Log()
        model, selection = _lazy_commitment(log)
        adapter = VisionModelAdapter(model)
        benchmark = _legacy_benchmark()
        check_memory(adapter, benchmark)
        assert selection.searches == [('search', 'IT')]
        assert ('activations', ('l2',), 1) in log  # the probe itself: one stimulus
        benchmark(LegacyBenchmarkFacade(adapter, visual_degrees=8))
        assert selection.searches == [('search', 'IT')]


class TestOtherMapsUnchanged:

    def test_committed_map_declares_its_own_regions(self):
        log = _Log()
        model = ModelCommitment('lazy-regions-test-no-such-plugin', _ActivationsModel(log), layers=['l1', 'l2'],
                                region_layer_map={'V4': 'l1', 'IT': 'l2'})
        adapter = VisionModelAdapter(model)
        assert adapter.out_channels == {'neural:V4', 'neural:IT', 'behavior'}
        assert adapter.region_layer_map == {'V4': 'l1', 'IT': 'l2'}
        assert adapter.probe_safe_regions == ['V4', 'IT']
        assert check_compatibility(adapter, _legacy_benchmark()).kind == 'full'
        neural_response(adapter, _stimuli(), ['neural:IT'])
        assert [entry for entry in log if entry[0] == 'search'] == []

    def test_plain_dict_map_declares_its_own_regions(self):
        adapter = VisionModelAdapter(_make_legacy_model(region_layer_map={'IT': 'layer4'}))
        assert adapter.out_channels == {'neural:IT', 'behavior'}

    def test_model_without_a_layer_model_declares_no_neural_output(self):
        adapter = VisionModelAdapter(_make_legacy_model())
        assert adapter.out_channels == {'behavior'}
        with pytest.raises(CompatibilityError):
            check_compatibility(adapter, _legacy_benchmark())
