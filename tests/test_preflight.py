"""Tests for pre-flight checks wired into the vision scoring pipeline."""

import pytest
from unittest.mock import patch, MagicMock

from brainscore_core.compatibility import CompatibilityError
from brainscore_core.io_catalog import modalities_to_input_channels
from brainscore_core.memory import MemoryError
from brainscore_core.contract import Subject


class _Subject(Subject):
    """A small real subject: it has what it declares and nothing else.

    A MagicMock answers every attribute, so a pre-flight that starts to read a
    new declaration would pass on mock values; this subject does not.
    """
    identifier = 'test-model'
    in_channels = frozenset()
    out_channels = frozenset()
    required_channels = frozenset()
    region_layer_map = {}

    def __init__(self, modalities):
        self.in_channels = modalities_to_input_channels(modalities)

    def participate(self, session):
        raise AssertionError('the stand-in benchmark never runs a session')


class TestPreflightInRunScore:
    """Verify that _run_score calls check_compatibility and check_memory."""

    def _make_model_and_benchmark(self, model_modalities, bench_required):
        """A real subject and a stand-in benchmark for pre-flight testing."""
        model = _Subject(model_modalities)

        benchmark = MagicMock(spec=['identifier', 'required_modalities', '__call__', 'preallocate_memory'])
        benchmark.identifier = 'test-bench'
        benchmark.required_modalities = bench_required
        return model, benchmark

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_incompatible_model_raises_compatibility_error(
        self, mock_load_model, mock_load_benchmark
    ):
        from brainscore_vision import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'text'},
            bench_required={'vision'},
        )
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        with pytest.raises(CompatibilityError, match="takes in none of the inputs"):
            _run_score('test-model', 'test-bench', check_mem=False)

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_score_metadata_attrs_are_stamped(
        self, mock_load_model, mock_load_benchmark
    ):
        from brainscore_vision import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        model.out_channels = {'neural:IT'}
        model.region_layer_map = {'IT': 'it-layer'}
        benchmark.region = 'IT'
        score = MagicMock()
        score.attrs = {'existing': 'preserved'}
        benchmark.return_value = score
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        result = _run_score('test-model', 'test-bench', check_mem=False)

        benchmark.assert_called_once_with(model)   # a compatible pair runs once
        assert result is score
        assert score.attrs['existing'] == 'preserved'
        assert score.attrs['in_channels'] == ('vision:image',)
        assert score.attrs['out_channels'] == ('neural:IT',)
        assert score.attrs['protocol'] == 'neural'
        assert score.attrs['subject_id'] == 'test-model'
        assert score.attrs['compatibility']['match'] == 'full'
        assert score.attrs['harness_id'] == 'brainscore_vision'

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_behavior_protocol_comes_from_recorded_measures(
            self, mock_load_model, mock_load_benchmark):
        from brainscore_vision import _run_score
        from brainscore_core.streaming import Session, StreamEvent
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            BehavioralAssembly,
        )

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        model.out_channels = {'behavior'}
        benchmark.parent = 'behavior'
        benchmark.requested_output_channels = {'behavior'}
        score = MagicMock()
        score.attrs = {}

        def run(candidate):
            # A subject's behavioral response, accepted by a session.
            Session((), ['behavior']).respond(
                StreamEvent('behavior', BehavioralAssembly(
                    [[1.0, 0.0]], dims=['presentation', 'choice'],
                    attrs={'measure': {'kind': 'reported', 'source': 'reply'}}), None))
            return score

        benchmark.side_effect = run
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        _run_score('test-model', 'test-bench', check_mem=False)

        assert score.attrs['protocol'] == 'behavior:generation'
        assert score.attrs['measure'] == 'reported'

    def test_an_older_core_fails_with_a_clear_message(self):
        from brainscore_vision import _run_score
        import brainscore_core.score_metadata as score_metadata

        saved = score_metadata.record_measures
        del score_metadata.record_measures
        try:
            with pytest.raises(ImportError, match='core ad7288a or later'):
                _run_score('test-model', 'test-bench', check_mem=False)
        finally:
            score_metadata.record_measures = saved

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_channel_mismatch_raises_channel_named_error(
        self, mock_load_model, mock_load_benchmark
    ):
        from brainscore_vision import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        benchmark.required_input_channels = {'text'}
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        with pytest.raises(CompatibilityError, match="text"):
            _run_score('test-model', 'test-bench', check_mem=False)

        benchmark.assert_not_called()

    @patch('brainscore_core.memory.check_memory')
    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_check_mem_false_skips_memory_check(
        self, mock_load_model, mock_load_benchmark, mock_check_memory
    ):
        from brainscore_vision import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        score = MagicMock()
        score.attrs = {}
        benchmark.return_value = score
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        _run_score('test-model', 'test-bench', check_mem=False)
        mock_check_memory.assert_not_called()

    @patch('brainscore_core.memory.check_memory')
    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_memory_error_propagates(
        self, mock_load_model, mock_load_benchmark, mock_check_memory
    ):
        from brainscore_vision import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark
        mock_check_memory.side_effect = MemoryError("not enough memory")

        with pytest.raises(MemoryError, match="not enough memory"):
            _run_score('test-model', 'test-bench', check_mem=True)

    def test_load_benchmark_sets_legacy_vision_required_modalities(self):
        import brainscore_vision

        benchmark = MagicMock()
        del benchmark.required_modalities
        del benchmark.accepted_modalities
        del benchmark.required_input_channels

        with patch.object(brainscore_vision, 'benchmark_registry',
                          {'legacy-bench': lambda: benchmark}):
            with patch('brainscore_vision.import_plugin'):
                loaded = brainscore_vision.load_benchmark('legacy-bench')

        assert loaded.required_modalities == {'vision'}
