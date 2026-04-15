"""
Integration tests for the pre-flight memory check (preallocate_memory).

Uses object.__new__ to bypass NeuralBenchmark.__init__ / timebins_from_assembly
so we can construct minimal benchmark fixtures without real S3 data.

Model is mocked at the BrainModel level: look_at returns a tiny xarray
DataArray with a 'neuroid' dim so the probe can read sizes['neuroid'].
place_on_screen short-circuits when source == target visual degrees (no I/O).
"""

import os
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
    _BYTES_PER_ELEMENT,
    preallocate_memory,
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


def _make_model(num_features: int = 512) -> BrainModel:
    """Mock BrainModel whose look_at returns a DataArray with neuroid dim."""
    model = MagicMock(spec=BrainModel)
    model.visual_degrees.return_value = _VISUAL_DEGREES

    def _look_at(stimuli, number_of_trials=1):
        n = len(stimuli)
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

    def test_raises_type_error_for_unknown_benchmark(self):
        class WeirdBenchmark:
            pass

        model = _make_model()
        with self.assertRaises(TypeError):
            preallocate_memory(model, WeirdBenchmark())


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


if __name__ == '__main__':
    unittest.main()
