"""Tests for pre-flight checks wired into the vision scoring pipeline."""

import pytest
from unittest.mock import patch, MagicMock

from brainscore_core.compatibility import (
    CompatibilityError,
    check_channel_compatibility,
    check_compatibility,
)
from brainscore_core.io_catalog import modalities_to_input_channels
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
        model.in_channels = modalities_to_input_channels(model_modalities)
        model.out_channels = set()
        model.required_channels = set()

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

    @patch('brainscore_core.compatibility.check_channel_compatibility')
    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_channel_check_runs_for_compatible_pair(
        self, mock_load_model, mock_load_benchmark, mock_check_channel
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

        mock_check_channel.assert_called_once_with(model, benchmark)
        benchmark.assert_called_once_with(model)

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

    @pytest.mark.parametrize(
        "model_id,benchmark_id,region",
        [
            ('alexnet', 'MajajHong2015.IT-pls', 'IT'),
            ('alexnet', 'MajajHong2015.V4-pls', 'V4'),
            ('alexnet', 'Allen2022_fmri_surface.IT-rdm', 'IT'),
            ('alexnet', 'Rajalingham2018-i2n', None),
            ('alexnet', 'Ferguson2024color-value_delta', None),
            ('hmax', 'MajajHong2015.IT-pls', 'IT'),
            ('hmax', 'MajajHong2015.V4-pls', 'V4'),
            ('hmax', 'Rajalingham2018-i2n', None),
            ('pixels', 'MajajHong2015.IT-pls', 'IT'),
            ('pixels', 'MajajHong2015.V4-pls', 'V4'),
            ('compact_vgg19_V4', 'MajajHong2015.IT-pls', 'IT'),
            ('compact_vgg19_V4', 'MajajHong2015.V4-pls', 'V4'),
            ('compact_vgg19_V4', 'Rajalingham2018-i2n', None),
            ('compact_vgg19_V4', 'Ferguson2024color-value_delta', None),
        ],
    )
    def test_m3_pairs_pass_modality_and_channel_preflight(
        self, model_id, benchmark_id, region
    ):
        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        model.identifier = model_id
        model.region_layer_map = {'IT': 'it-layer', 'V4': 'v4-layer'}
        model.out_channels = {'neural:IT', 'neural:V4'}
        benchmark.identifier = benchmark_id
        if region is not None:
            benchmark.region = region

        check_compatibility(model, benchmark)
        check_channel_compatibility(model, benchmark)

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

    def test_load_benchmark_sets_legacy_vision_required_modalities(self):
        import brainscore_vision

        benchmark = MagicMock()
        del benchmark.required_modalities
        del benchmark.accepted_modalities

        with patch.object(brainscore_vision, 'benchmark_registry',
                          {'legacy-bench': lambda: benchmark}):
            with patch('brainscore_vision.import_plugin'):
                loaded = brainscore_vision.load_benchmark('legacy-bench')

        assert loaded.required_modalities == {'vision'}
