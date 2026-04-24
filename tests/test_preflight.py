"""Tests for pre-flight checks wired into the vision scoring pipeline."""

import pytest
from unittest.mock import patch, MagicMock

from brainscore_core.compatibility import CompatibilityError
from brainscore_core.memory import MemoryError


class TestPreflightInRunScore:
    """Verify that _run_score calls check_compatibility and check_memory."""

    def _make_model_and_benchmark(self, model_modalities, bench_required):
        """Create mock model and benchmark for pre-flight testing."""
        model = MagicMock()
        model.identifier = 'test-model'
        # Two-tier model-side modality declaration. Legacy tests set only
        # supported_modalities; mirror that into available_modalities + an
        # empty required_modalities so the pre-flight checker reads a
        # coherent pair of sets rather than MagicMock auto-attrs.
        model.supported_modalities = model_modalities
        model.available_modalities = model_modalities
        model.required_modalities = set()
        model.region_layer_map = {}

        benchmark = MagicMock(spec=['identifier', 'required_modalities', '__call__'])
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

        with pytest.raises(CompatibilityError, match="does not support modalities required"):
            _run_score('test-model', 'test-bench', check_mem=False)

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_compatible_model_proceeds_to_scoring(
        self, mock_load_model, mock_load_benchmark
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

        result = _run_score('test-model', 'test-bench', check_mem=False)
        benchmark.assert_called_once_with(model)

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
    def test_check_mem_true_calls_memory_check(
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

        _run_score('test-model', 'test-bench', check_mem=True)
        mock_check_memory.assert_called_once_with(model, benchmark)

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
