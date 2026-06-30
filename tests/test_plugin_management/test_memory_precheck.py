"""
Integration tests for the pre-flight memory check (preallocate_memory).

Uses object.__new__ to bypass NeuralBenchmark.__init__ / timebins_from_assembly
so we can construct minimal benchmark fixtures without real S3 data.

Model is mocked at the BrainModel level: look_at returns a tiny xarray
DataArray with a 'neuroid' dim so the probe can read sizes['neuroid'].
place_on_screen short-circuits when source == target visual degrees (no I/O).
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr
import pytest

from brainscore_core import Score
from brainscore_core.benchmarks import score_benchmark
from brainscore_vision.benchmark_helpers.memory import (
    MemoryEstimate,
    _OVERHEAD_FACTOR,
    _PLS_OVERHEAD_FACTOR,
    _BYTES_PER_ELEMENT,
    _DEFAULT_CALIBRATION_PATH,
    preallocate_memory,
    load_calibration,
    save_calibration,
)
from brainscore_vision.benchmark_helpers.neural_common import (
    NeuralBenchmark,
    TrainTestNeuralBenchmark,
)
from brainscore_vision.model_interface import BrainModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VISUAL_DEGREES = 8   # source == target so place_on_screen is a no-op


def _make_stimulus_set(n: int = 10):
    """Minimal StimulusSet-like DataFrame with stimulus_id coordinate."""
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    import pandas as pd
    df = pd.DataFrame({'stimulus_id': [f'img{i:03d}' for i in range(n)],
                       'image_file_name': [f'img{i:03d}.png' for i in range(n)]})
    ss = StimulusSet(df)
    ss.identifier = 'test_stimulus_set'
    ss.stimulus_paths = {row.stimulus_id: f'/tmp/{row.image_file_name}'
                         for _, row in df.iterrows()}
    return ss


def _make_neural_benchmark(n_stimuli: int = 10, n_trials: int = 1,
                            timebins=None, region: str = 'IT') -> NeuralBenchmark:
    """Construct a NeuralBenchmark without calling __init__."""
    bm = object.__new__(NeuralBenchmark)
    bm._identifier = 'test-neural-benchmark'
    bm._number_of_trials = n_trials
    bm.timebins = timebins or [(70, 170)]
    bm.region = region
    bm._visual_degrees = _VISUAL_DEGREES
    bm._ceiling_func = lambda: Score(0.8)

    ss = _make_stimulus_set(n_stimuli)
    assembly = MagicMock()
    assembly.stimulus_set = ss
    bm._assembly = assembly
    return bm


def _make_train_test_benchmark(n_train: int = 8, n_test: int = 4) -> TrainTestNeuralBenchmark:
    """Construct a TrainTestNeuralBenchmark without calling __init__."""
    bm = object.__new__(TrainTestNeuralBenchmark)
    bm._identifier = 'test-train-test-benchmark'
    bm._number_of_trials = 1
    bm.timebins = [(70, 170)]
    bm.region = 'IT'
    bm._visual_degrees = _VISUAL_DEGREES
    bm._ceiling_func = lambda: Score(0.8)

    train_assembly = MagicMock()
    train_assembly.stimulus_set = _make_stimulus_set(n_train)
    test_assembly = MagicMock()
    test_assembly.stimulus_set = _make_stimulus_set(n_test)
    bm.train_assembly = train_assembly
    bm.test_assembly = test_assembly
    return bm


def _make_model(num_features: int = 512, num_timebins: int = 1) -> BrainModel:
    """Mock BrainModel whose look_at returns a DataArray with neuroid (and
    optionally time_bin) dim. ``num_timebins > 1`` mimics a temporal model so
    the preflight's ``probe_output.sizes.get('time_bin', 1)`` path picks up
    the right count."""
    model = MagicMock(spec=BrainModel)
    model.visual_degrees.return_value = _VISUAL_DEGREES

    def _look_at(stimuli, number_of_trials=1):
        n = len(stimuli)
        if num_timebins > 1:
            data = np.zeros((n, num_features, num_timebins))
            return xr.DataArray(
                data,
                dims=['presentation', 'neuroid', 'time_bin'],
                coords={
                    'stimulus_id': ('presentation', stimuli['stimulus_id'].values),
                    'neuroid_id': ('neuroid', np.arange(num_features)),
                    'time_bin': np.arange(num_timebins),
                },
            )
        data = np.zeros((n, num_features))
        return xr.DataArray(
            data,
            dims=['presentation', 'neuroid'],
            coords={
                'stimulus_id': ('presentation', stimuli['stimulus_id'].values),
                'neuroid_id': ('neuroid', np.arange(num_features)),
            },
        )

    model.look_at.side_effect = _look_at
    model.activations_model = None   # no LayerPCA
    return model


# ---------------------------------------------------------------------------
# TestMemoryEstimateShape
# ---------------------------------------------------------------------------

class TestMemoryEstimateShape(unittest.TestCase):

    def setUp(self):
        self.bm = _make_neural_benchmark(n_stimuli=10)
        self.model = _make_model(num_features=512)

    def test_estimate_fields(self):
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 32 * (1024 ** 3)
            est = preallocate_memory(self.model, self.bm, raise_if_oom=False)

        self.assertEqual(est.num_stimuli, 10)
        self.assertEqual(est.num_features, 512)
        self.assertEqual(est.num_timebins, 1)

    def test_activation_gb_formula(self):
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 32 * (1024 ** 3)
            est = preallocate_memory(self.model, self.bm, raise_if_oom=False)

        expected_bytes = 10 * 512 * 1 * _BYTES_PER_ELEMENT
        expected_gb = expected_bytes / (1024 ** 3)
        self.assertAlmostEqual(est.activation_gb, expected_gb, places=6)

    def test_total_estimated_gb(self):
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 32 * (1024 ** 3)
            est = preallocate_memory(self.model, self.bm, raise_if_oom=False)

        self.assertAlmostEqual(est.total_estimated_gb,
                               est.activation_gb * _OVERHEAD_FACTOR, places=6)

    def test_available_gb_from_psutil(self):
        available_bytes = 16 * (1024 ** 3)
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = available_bytes
            est = preallocate_memory(self.model, self.bm, raise_if_oom=False)

        self.assertAlmostEqual(est.available_gb, 16.0, places=3)


# ---------------------------------------------------------------------------
# TestOOMDetection
# ---------------------------------------------------------------------------

class TestOOMDetection(unittest.TestCase):

    def _estimate(self, available_gb, num_features=1_000_000, n_stimuli=100):
        bm = _make_neural_benchmark(n_stimuli=n_stimuli)
        model = _make_model(num_features=num_features)
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = int(available_gb * (1024 ** 3))
            return preallocate_memory(model, bm, raise_if_oom=False)

    def test_will_oom_true_when_over(self):
        est = self._estimate(available_gb=0.001)
        self.assertTrue(est.will_oom)

    def test_will_oom_false_when_under(self):
        est = self._estimate(available_gb=1000)
        self.assertFalse(est.will_oom)

    def test_raises_memory_error_when_raise_if_oom(self):
        bm = _make_neural_benchmark(n_stimuli=100)
        model = _make_model(num_features=1_000_000)
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 1
            with self.assertRaises(MemoryError):
                preallocate_memory(model, bm, raise_if_oom=True)

    def test_no_raise_when_raise_if_oom_false(self):
        bm = _make_neural_benchmark(n_stimuli=100)
        model = _make_model(num_features=1_000_000)
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 1
            est = preallocate_memory(model, bm, raise_if_oom=False)
        self.assertTrue(est.will_oom)


# ---------------------------------------------------------------------------
# TestProbeUsesOneStimulusOnly
# ---------------------------------------------------------------------------

class TestProbeUsesOneStimulusOnly(unittest.TestCase):

    def test_look_at_called_with_one_stimulus(self):
        bm = _make_neural_benchmark(n_stimuli=100)
        model = _make_model()
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 32 * (1024 ** 3)
            preallocate_memory(model, bm, raise_if_oom=False)

        call_args = model.look_at.call_args
        stimuli_arg = call_args[0][0]
        self.assertEqual(len(stimuli_arg), 1)

    def test_num_stimuli_reflects_full_benchmark_not_probe(self):
        bm = _make_neural_benchmark(n_stimuli=42)
        model = _make_model()
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 32 * (1024 ** 3)
            est = preallocate_memory(model, bm, raise_if_oom=False)

        self.assertEqual(est.num_stimuli, 42)


# ---------------------------------------------------------------------------
# TestScoreBenchmarkAbortOnOOM
# ---------------------------------------------------------------------------

class TestScoreBenchmarkAbortOnOOM(unittest.TestCase):

    def test_score_benchmark_aborts_before_calling_benchmark(self):
        """score_benchmark should raise MemoryError before __call__ is invoked."""
        bm = _make_neural_benchmark(n_stimuli=100)
        bm.__call__ = MagicMock(return_value=Score(0.5))
        model = _make_model(num_features=1_000_000)

        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 1
            with self.assertRaises(MemoryError):
                score_benchmark(bm, model)

        bm.__call__.assert_not_called()

    def test_score_benchmark_calls_benchmark_when_ok(self):
        bm = _make_neural_benchmark(n_stimuli=5)
        score_val = Score(0.42)
        score_val.attrs['ceiling'] = Score(1.0)
        model = _make_model(num_features=10)

        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 32 * (1024 ** 3)
            with patch.object(NeuralBenchmark, '__call__', return_value=score_val) as mock_call:
                result = score_benchmark(bm, model)

        mock_call.assert_called_once_with(model)
        self.assertEqual(float(result), 0.42)


# ---------------------------------------------------------------------------
# TestSkipEnvVar
# ---------------------------------------------------------------------------

class TestSkipEnvVar(unittest.TestCase):

    def test_returns_none_when_env_var_set(self):
        bm = _make_neural_benchmark()
        model = _make_model()
        with patch.dict(os.environ, {'BRAINSCORE_SKIP_MEMORY_CHECK': '1'}):
            result = preallocate_memory(model, bm, raise_if_oom=True)
        self.assertIsNone(result)

    def test_runs_normally_when_env_var_unset(self):
        bm = _make_neural_benchmark()
        model = _make_model()
        with patch.dict(os.environ, {'BRAINSCORE_SKIP_MEMORY_CHECK': '0'}):
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value.available = 32 * (1024 ** 3)
                result = preallocate_memory(model, bm, raise_if_oom=False)
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# TestUnsupportedBenchmarkType
# ---------------------------------------------------------------------------

class TestUnsupportedBenchmarkType(unittest.TestCase):

    def test_returns_none_for_unknown_benchmark(self):
        # Unsupported benchmark types (e.g. behavioral/engineering called
        # directly) skip preflight and return None rather than crashing.
        class WeirdBenchmark:
            pass

        model = _make_model()
        self.assertIsNone(preallocate_memory(model, WeirdBenchmark()))

    def test_unsupported_benchmark_emits_skipped_sentinel(self):
        """Behavioral / engineering / wrapper benchmarks return None from
        preallocate_memory because they don't have an activation-based
        formula. They must still emit a BRAINSCORE_PREFLIGHT sentinel with
        ``formula_type='skipped'`` so the downstream ResourceUsage row
        records that preflight WAS attempted on this run -- distinguishing
        it from older container images where preflight was never wired up
        (which leaves the column NULL). Surfaced in infrastructure #48
        diagnosis where 17/26 production rows had a NULL
        preflight_formula_type for exactly this reason."""
        class BehavioralBenchmark:
            identifier = 'Geirhos2021edge-error_consistency'

        model = _make_model()
        with patch('builtins.print') as mock_print:
            result = preallocate_memory(model, BehavioralBenchmark())

        self.assertIsNone(result)
        # The sentinel goes to stdout via print() -- inspect the captured calls.
        sentinel_lines = [
            args[0] for (args, kwargs) in mock_print.call_args_list
            if args and isinstance(args[0], str)
            and args[0].startswith('BRAINSCORE_PREFLIGHT')
        ]
        self.assertEqual(len(sentinel_lines), 1,
                         f"expected one preflight sentinel, got {sentinel_lines}")
        payload = json.loads(sentinel_lines[0].split(' ', 1)[1])
        self.assertEqual(payload['formula_type'], 'skipped')
        self.assertEqual(payload['reason'], 'unsupported_benchmark_type')
        self.assertEqual(payload['benchmark_type'], 'BehavioralBenchmark')
        self.assertIsNone(payload['estimate_gb'])
        self.assertFalse(payload['will_oom'])

    def test_env_skip_also_emits_skipped_sentinel(self):
        """BRAINSCORE_SKIP_MEMORY_CHECK=1 short-circuits preflight, but the
        sentinel still goes out so the row records that the path was
        traversed and intentionally bypassed."""
        bm = _make_neural_benchmark()
        model = _make_model()
        with patch.dict(os.environ, {'BRAINSCORE_SKIP_MEMORY_CHECK': '1'}):
            with patch('builtins.print') as mock_print:
                result = preallocate_memory(model, bm, raise_if_oom=False)
        self.assertIsNone(result)
        sentinel_lines = [
            args[0] for (args, kwargs) in mock_print.call_args_list
            if args and isinstance(args[0], str)
            and args[0].startswith('BRAINSCORE_PREFLIGHT')
        ]
        self.assertEqual(len(sentinel_lines), 1)
        payload = json.loads(sentinel_lines[0].split(' ', 1)[1])
        self.assertEqual(payload['formula_type'], 'skipped')
        self.assertEqual(payload['reason'], 'env_skip')


# ---------------------------------------------------------------------------
# TestTrainTestNeuralBenchmark
# ---------------------------------------------------------------------------

class TestTrainTestNeuralBenchmark(unittest.TestCase):

    def test_num_stimuli_is_train_plus_test(self):
        bm = _make_train_test_benchmark(n_train=8, n_test=4)
        model = _make_model()
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 32 * (1024 ** 3)
            est = preallocate_memory(model, bm, raise_if_oom=False)

        self.assertEqual(est.num_stimuli, 12)

    def test_estimate_formula_train_test(self):
        bm = _make_train_test_benchmark(n_train=8, n_test=4)
        model = _make_model(num_features=256)
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 32 * (1024 ** 3)
            est = preallocate_memory(model, bm, raise_if_oom=False)

        expected_bytes = 12 * 256 * 1 * _BYTES_PER_ELEMENT
        self.assertAlmostEqual(est.activation_gb, expected_bytes / (1024 ** 3), places=6)


# ---------------------------------------------------------------------------
# TestCalibrationIO  —  load_calibration / save_calibration
# ---------------------------------------------------------------------------

class TestCalibrationIO(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._cal_path = os.path.join(self._tmpdir, 'benchmark_costs.json')

    def test_load_returns_empty_dict_when_file_missing(self):
        result = load_calibration('/nonexistent/path/benchmark_costs.json')
        self.assertEqual(result, {})

    def test_save_and_load_roundtrip(self):
        costs = {'MajajHong2015.IT-pls': 2.8336, 'Allen2022_fmri.V1-ridge': 0.5544}
        save_calibration(costs, self._cal_path)
        loaded = load_calibration(self._cal_path)
        self.assertEqual(loaded, costs)

    def test_save_creates_intermediate_directories(self):
        deep_path = os.path.join(self._tmpdir, 'a', 'b', 'c', 'costs.json')
        save_calibration({'bm': 1.0}, deep_path)
        self.assertTrue(os.path.exists(deep_path))

    def test_load_handles_corrupt_file_gracefully(self):
        with open(self._cal_path, 'w') as f:
            f.write('not valid json {{{')
        result = load_calibration(self._cal_path)
        self.assertEqual(result, {})

    def test_save_writes_valid_json(self):
        costs = {'foo-bar': 3.14}
        save_calibration(costs, self._cal_path)
        with open(self._cal_path) as f:
            data = json.load(f)
        self.assertAlmostEqual(data['foo-bar'], 3.14)

    def test_save_overwrites_existing_file(self):
        save_calibration({'old': 1.0}, self._cal_path)
        save_calibration({'new': 2.0}, self._cal_path)
        loaded = load_calibration(self._cal_path)
        self.assertNotIn('old', loaded)
        self.assertAlmostEqual(loaded['new'], 2.0)


# ---------------------------------------------------------------------------
# TestCalibratedFormula  —  two-component formula vs ×6 fallback
# ---------------------------------------------------------------------------

class TestCalibratedFormula(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._cal_path = os.path.join(self._tmpdir, 'costs.json')

    def _estimate(self, bm, model, fixed_cost=None, cal_path=None):
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 64 * (1024 ** 3)
            with patch('brainscore_vision.benchmark_helpers.memory._DEFAULT_CALIBRATION_PATH',
                       cal_path or '/nonexistent'):
                return preallocate_memory(model, bm, raise_if_oom=False,
                                          fixed_benchmark_cost_gb=fixed_cost)

    def test_explicit_fixed_cost_overrides_fallback(self):
        bm = _make_neural_benchmark(n_stimuli=10)
        model = _make_model(num_features=512)
        est = self._estimate(bm, model, fixed_cost=5.0)
        self.assertAlmostEqual(est.total_estimated_gb, est.activation_gb + 5.0, places=5)

    def test_fixed_cost_stored_in_estimate(self):
        bm = _make_neural_benchmark(n_stimuli=10)
        model = _make_model(num_features=512)
        est = self._estimate(bm, model, fixed_cost=3.5)
        self.assertAlmostEqual(est.fixed_benchmark_cost_gb, 3.5)

    def test_falls_back_to_overhead_when_no_calibration(self):
        bm = _make_neural_benchmark(n_stimuli=10)
        model = _make_model(num_features=512)
        est = self._estimate(bm, model, fixed_cost=None, cal_path='/nonexistent')
        self.assertIsNone(est.fixed_benchmark_cost_gb)
        self.assertAlmostEqual(est.total_estimated_gb,
                               est.activation_gb * _OVERHEAD_FACTOR, places=5)

    def test_auto_loads_fixed_cost_from_calibration_json(self):
        bm = _make_neural_benchmark(n_stimuli=10)
        bm._identifier = 'MajajHong2015.IT-pls'
        model = _make_model(num_features=512)
        save_calibration({'MajajHong2015.IT-pls': 2.8336}, self._cal_path)

        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 64 * (1024 ** 3)
            with patch('brainscore_vision.benchmark_helpers.memory._DEFAULT_CALIBRATION_PATH',
                       self._cal_path):
                est = preallocate_memory(model, bm, raise_if_oom=False)

        self.assertAlmostEqual(est.fixed_benchmark_cost_gb, 2.8336, places=4)
        self.assertAlmostEqual(est.total_estimated_gb,
                               est.activation_gb * _PLS_OVERHEAD_FACTOR + 2.8336, places=4)

    def test_benchmark_not_in_table_uses_fallback(self):
        bm = _make_neural_benchmark(n_stimuli=10)
        bm._identifier = 'unknown-benchmark'
        model = _make_model(num_features=512)
        save_calibration({'MajajHong2015.IT-pls': 2.8336}, self._cal_path)

        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 64 * (1024 ** 3)
            with patch('brainscore_vision.benchmark_helpers.memory._DEFAULT_CALIBRATION_PATH',
                       self._cal_path):
                est = preallocate_memory(model, bm, raise_if_oom=False)

        self.assertIsNone(est.fixed_benchmark_cost_gb)
        self.assertAlmostEqual(est.total_estimated_gb,
                               est.activation_gb * _OVERHEAD_FACTOR, places=5)

    def test_temporal_pls_scales_estimate_by_num_timebins(self):
        """Temporal PLS retains per-timebin intermediates that the regular
        PLS overhead formula doesn't account for, so peak memory grows
        linearly with the actual number of timebins probed. A fixed constant
        is wrong because temporal benchmarks may have anywhere from 2 to
        dozens of timebins — scale by what we measure at preflight time.

        Driven by infrastructure #48 (MajajHong2015public.{V4,IT}-temporal-pls
        OOMing on small tier despite the preflight saying it would fit)."""
        bm = _make_neural_benchmark(n_stimuli=10)
        bm._identifier = 'MajajHong2015public.IT-temporal-pls'
        # Mock model returns a probe output with 10 timebins so the preflight
        # picks ``num_timebins=10`` via probe_output.sizes['time_bin'].
        model = _make_model(num_features=512, num_timebins=10)
        save_calibration({'MajajHong2015public.IT-temporal-pls': 1.5}, self._cal_path)
        est = self._estimate(bm, model, cal_path=self._cal_path)

        # activation_gb already accounts for the 10 timebins via the bytes
        # formula. The temporal-pls correction multiplies the *full* PLS
        # estimate by num_timebins on top of that.
        expected_pls = est.activation_gb * _PLS_OVERHEAD_FACTOR + 1.5
        expected_total = expected_pls * 10
        self.assertAlmostEqual(est.total_estimated_gb, expected_total, places=4)
        self.assertEqual(est.formula_type, 'temporal_pls')
        self.assertEqual(est.num_timebins, 10)

        # Different timebin count → different multiplier. A 3-timebin probe
        # produces a 3× multiplier, not 10×.
        bm3 = _make_neural_benchmark(n_stimuli=10)
        bm3._identifier = 'MajajHong2015public.V4-temporal-pls'
        model3 = _make_model(num_features=512, num_timebins=3)
        save_calibration({'MajajHong2015public.V4-temporal-pls': 1.5}, self._cal_path)
        est3 = self._estimate(bm3, model3, cal_path=self._cal_path)
        expected3 = (est3.activation_gb * _PLS_OVERHEAD_FACTOR + 1.5) * 3
        self.assertAlmostEqual(est3.total_estimated_gb, expected3, places=4)
        self.assertEqual(est3.num_timebins, 3)

    def test_pls_single_timebin_skips_temporal_multiplier(self):
        """Single-window PLS benchmarks (MajajHong2015.IT-pls etc.) must
        stay on the regular PLS formula. The temporal multiplier only fires
        when ``num_timebins > 1`` so single-timebin PLS is unchanged."""
        bm = _make_neural_benchmark(n_stimuli=10)
        bm._identifier = 'MajajHong2015.IT-pls'
        model = _make_model(num_features=512, num_timebins=1)
        est = self._estimate(bm, model, fixed_cost=1.5)
        self.assertEqual(est.formula_type, 'pls')
        self.assertAlmostEqual(
            est.total_estimated_gb,
            est.activation_gb * _PLS_OVERHEAD_FACTOR + 1.5,
            places=5,
        )

    def test_oom_detected_with_calibrated_formula(self):
        bm = _make_neural_benchmark(n_stimuli=10)
        model = _make_model(num_features=512)
        # fixed cost alone exceeds available RAM
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = int(0.001 * (1024 ** 3))
            with self.assertRaises(MemoryError):
                preallocate_memory(model, bm, raise_if_oom=True, fixed_benchmark_cost_gb=100.0)


# ---------------------------------------------------------------------------
# TestMemoryEstimateStr  —  __str__ output
# ---------------------------------------------------------------------------

class TestMemoryEstimateStr(unittest.TestCase):

    def _make_estimate(self, fixed_cost=None, will_oom=False):
        available = 1.0 if will_oom else 100.0
        total = 200.0 if will_oom else 1.5
        # MemoryEstimate.__str__ picks the formula label off formula_type, not
        # off fixed_benchmark_cost_gb. Mirror what preallocate_memory does:
        # calibrated when a fixed cost is supplied, fallback otherwise.
        formula_type = 'calibrated' if fixed_cost is not None else 'fallback'
        return MemoryEstimate(
            num_stimuli=100,
            num_trials=1,
            num_features=512,
            num_timebins=1,
            activation_gb=0.5,
            total_estimated_gb=total,
            available_gb=available,
            fixed_benchmark_cost_gb=fixed_cost,
            formula_type=formula_type,
        )

    def test_str_shows_ok_when_not_oom(self):
        est = self._make_estimate()
        self.assertIn('[OK]', str(est))

    def test_str_shows_oom_likely_when_oom(self):
        est = self._make_estimate(will_oom=True)
        self.assertIn('[OOM LIKELY]', str(est))

    def test_str_shows_calibrated_formula_when_fixed_cost_set(self):
        est = self._make_estimate(fixed_cost=3.5)
        s = str(est)
        self.assertIn('fixed benchmark cost', s)
        self.assertNotIn(f'×{_OVERHEAD_FACTOR}', s)

    def test_str_shows_overhead_formula_when_no_fixed_cost(self):
        est = self._make_estimate(fixed_cost=None)
        s = str(est)
        self.assertIn(f'×{_OVERHEAD_FACTOR}', s)
        self.assertNotIn('fixed benchmark cost', s)

    def test_str_contains_stimuli_and_features(self):
        est = self._make_estimate()
        s = str(est)
        self.assertIn('100', s)   # num_stimuli
        self.assertIn('512', s)   # num_features


# ---------------------------------------------------------------------------
# TestCalibratedIntegration  —  full pipeline with a real JSON file
# ---------------------------------------------------------------------------

class TestCalibratedIntegration(unittest.TestCase):
    """
    End-to-end test: save a calibration JSON, then verify preallocate_memory
    picks it up automatically and produces the correct two-component estimate.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._cal_path = os.path.join(self._tmpdir, 'costs.json')

    def test_full_roundtrip_calibrated_estimate(self):
        n_stimuli = 20
        n_features = 256
        fixed_cost = 4.35

        bm = _make_neural_benchmark(n_stimuli=n_stimuli)
        bm._identifier = 'integration-test-benchmark'
        model = _make_model(num_features=n_features)

        save_calibration({'integration-test-benchmark': fixed_cost}, self._cal_path)

        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 64 * (1024 ** 3)
            with patch('brainscore_vision.benchmark_helpers.memory._DEFAULT_CALIBRATION_PATH',
                       self._cal_path):
                est = preallocate_memory(model, bm, raise_if_oom=False)

        expected_activation = n_stimuli * n_features * 1 * _BYTES_PER_ELEMENT / (1024 ** 3)
        self.assertAlmostEqual(est.activation_gb, expected_activation, places=6)
        self.assertAlmostEqual(est.fixed_benchmark_cost_gb, fixed_cost, places=4)
        self.assertAlmostEqual(est.total_estimated_gb, expected_activation + fixed_cost, places=4)
        self.assertFalse(est.will_oom)

    def test_score_benchmark_uses_preallocate_memory(self):
        """score_benchmark must call preallocate_memory before __call__."""
        bm = _make_neural_benchmark(n_stimuli=5)
        model = _make_model(num_features=10)
        score_val = MagicMock()

        call_order = []

        def _fake_preallocate(self, candidate):
            call_order.append('preallocate')

        def _fake_call(self, candidate):
            call_order.append('score')
            return score_val

        with patch.object(NeuralBenchmark, 'preallocate_memory', _fake_preallocate):
            with patch.object(NeuralBenchmark, '__call__', _fake_call):
                score_benchmark(bm, model)

        self.assertEqual(call_order, ['preallocate', 'score'])


if __name__ == '__main__':
    unittest.main()


class TestBenchmarkScaffoldingOverhead(unittest.TestCase):

    def _estimate(self, bm, model):
        from brainscore_vision.benchmark_helpers.memory import preallocate_memory
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 256 * (1024 ** 3)
            with patch('brainscore_vision.benchmark_helpers.memory._DEFAULT_CALIBRATION_PATH',
                       '/nonexistent'):
                return preallocate_memory(model, bm, raise_if_oom=False)

    def test_overhead_added_when_benchmark_in_table(self):
        bm = _make_neural_benchmark(n_stimuli=100)
        bm._identifier = 'Zerbe2026_fmri.V1-tau-ridgecv'
        model = _make_model(num_features=1024)
        with patch.dict(
                'brainscore_vision.benchmark_helpers.memory._BENCHMARK_SCAFFOLDING_OVERHEAD_GB',
                {'Zerbe2026_fmri.V1-tau-ridgecv': 42.0}, clear=False):
            est = self._estimate(bm, model)
        baseline = est.activation_gb * _OVERHEAD_FACTOR
        self.assertGreaterEqual(est.total_estimated_gb, baseline + 42.0 - 0.5)

    def test_overhead_not_applied_when_benchmark_missing(self):
        bm = _make_neural_benchmark(n_stimuli=10)
        bm._identifier = 'BenchmarkNotInTable-pls'
        model = _make_model(num_features=512)
        with patch.dict(
                'brainscore_vision.benchmark_helpers.memory._BENCHMARK_SCAFFOLDING_OVERHEAD_GB',
                {'Zerbe2026_fmri.V1-tau-ridgecv': 42.0}, clear=True):
            est = self._estimate(bm, model)
        self.assertAlmostEqual(est.total_estimated_gb,
                               est.activation_gb * _OVERHEAD_FACTOR, places=4)

    def test_zero_overhead_entry_is_no_op(self):
        bm = _make_neural_benchmark(n_stimuli=10)
        bm._identifier = 'BenchmarkWithZero-pls'
        model = _make_model(num_features=512)
        with patch.dict(
                'brainscore_vision.benchmark_helpers.memory._BENCHMARK_SCAFFOLDING_OVERHEAD_GB',
                {'BenchmarkWithZero-pls': 0.0}, clear=False):
            est = self._estimate(bm, model)
        self.assertAlmostEqual(est.total_estimated_gb,
                               est.activation_gb * _OVERHEAD_FACTOR, places=4)

    def test_production_table_has_only_positive_corrections(self):
        from brainscore_vision.benchmark_helpers.memory import _BENCHMARK_SCAFFOLDING_OVERHEAD_GB
        self.assertTrue(all(v > 0 for v in _BENCHMARK_SCAFFOLDING_OVERHEAD_GB.values()))

    def test_production_table_does_not_include_papale_v4_ridgecv(self):
        from brainscore_vision.benchmark_helpers.memory import _BENCHMARK_SCAFFOLDING_OVERHEAD_GB
        self.assertNotIn('Papale2025.V4-ridgecv', _BENCHMARK_SCAFFOLDING_OVERHEAD_GB)
