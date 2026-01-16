"""
Tests for the Synthetic Temporal Match-to-Sample Benchmark.
"""

import numpy as np
import pytest

from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision.model_interface import BrainModel


class MockTemporalActivationsModel:
    """Mock activations model that returns temporal features with a time_bin dimension."""

    def __init__(self, feature_map: dict = None):
        self.identifier = 'mock-temporal-model'
        self.feature_map = feature_map

    def __call__(self, stimuli, layers, require_variance=False):
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

        n_stimuli = len(stimuli)
        n_neuroids = 10
        n_time_bins = 5

        if self.feature_map is not None:
            features = np.array([self.feature_map.get(sid, np.random.rand(n_neuroids))
                                 for sid in stimuli['stimulus_id'].values])
            features = np.repeat(features[:, :, np.newaxis], n_time_bins, axis=2)
        else:
            np.random.seed(42)
            features = np.random.rand(n_stimuli, n_neuroids, n_time_bins)

        assembly = NeuroidAssembly(
            features,
            coords={
                'stimulus_id': ('presentation', list(stimuli['stimulus_id'].values)),
                'neuroid_id': ('neuroid', [f'neuroid_{i}' for i in range(n_neuroids)]),
                'time_bin_start': ('time_bin', [i * 100 for i in range(n_time_bins)]),
                'time_bin_end': ('time_bin', [(i + 1) * 100 for i in range(n_time_bins)]),
            },
            dims=['presentation', 'neuroid', 'time_bin']
        )
        return assembly


class MockPerfectTemporalModel:
    """Mock model that always picks the correct choice based on stimulus direction."""

    def __init__(self):
        self.identifier = 'mock-perfect-temporal'
        self._direction_to_features = {}

    def __call__(self, stimuli, layers, require_variance=False):
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

        n_stimuli = len(stimuli)
        n_neuroids = 10
        n_time_bins = 5

        features = []
        for _, row in stimuli.iterrows():
            direction = row.get('direction', 0)
            # Create unique feature vector for each direction
            feature = np.zeros(n_neuroids)
            direction_idx = int((direction / (np.pi / 4)) % 8)
            feature[direction_idx] = 1.0
            features.append(feature)

        features = np.array(features)
        features = np.repeat(features[:, :, np.newaxis], n_time_bins, axis=2)

        assembly = NeuroidAssembly(
            features,
            coords={
                'stimulus_id': ('presentation', list(stimuli['stimulus_id'].values)),
                'neuroid_id': ('neuroid', [f'neuroid_{i}' for i in range(n_neuroids)]),
                'time_bin_start': ('time_bin', [i * 100 for i in range(n_time_bins)]),
                'time_bin_end': ('time_bin', [(i + 1) * 100 for i in range(n_time_bins)]),
            },
            dims=['presentation', 'neuroid', 'time_bin']
        )
        return assembly


class TestStimuliGeneration:
    def test_generate_moving_dot_frames(self):
        from brainscore_vision.benchmarks.synthetic_temporal_matching.stimuli import (
            generate_moving_dot_frames
        )

        frames = generate_moving_dot_frames(
            direction=0,  # Moving right
            n_frames=16,
            frame_size=(64, 64),
            seed=42
        )

        assert frames.shape == (16, 64, 64, 3)
        assert frames.dtype == np.uint8
        # Should have some white pixels (the dot)
        assert np.sum(frames > 0) > 0

    def test_generate_synthetic_benchmark_stimuli(self):
        from brainscore_vision.benchmarks.synthetic_temporal_matching.stimuli import (
            generate_synthetic_benchmark_stimuli
        )

        stimuli, correct_choices = generate_synthetic_benchmark_stimuli(
            n_trials=5,
            n_choices=3,
            seed=42
        )

        # Should have 5 trials * (1 sample + 3 choices) = 20 stimuli
        assert len(stimuli) == 5 * 4
        assert len(correct_choices) == 5

        # Each correct choice should be valid index
        for choice in correct_choices:
            assert 0 <= choice < 3

        # Check stimulus structure
        for stim in stimuli:
            assert 'stimulus_id' in stim
            assert 'trial_id' in stim
            assert 'stimulus_role' in stim
            assert stim['stimulus_role'] in ['sample', 'choice']


class TestSyntheticBenchmark:
    def test_benchmark_creation(self):
        from brainscore_vision.benchmarks.synthetic_temporal_matching.benchmark import (
            SyntheticTemporalMatching
        )

        benchmark = SyntheticTemporalMatching(n_trials=5, n_choices=3, seed=42)

        assert benchmark.identifier == 'SyntheticTemporalMatching'
        assert benchmark._n_trials == 5
        assert benchmark._n_choices == 3

    def test_benchmark_stimulus_set(self):
        from brainscore_vision.benchmarks.synthetic_temporal_matching.benchmark import (
            SyntheticTemporalMatching
        )

        benchmark = SyntheticTemporalMatching(n_trials=5, n_choices=3, seed=42)
        stimuli = benchmark._stimulus_set

        assert isinstance(stimuli, StimulusSet)
        assert 'trial_id' in stimuli.columns
        assert 'stimulus_role' in stimuli.columns
        assert 'choice_index' in stimuli.columns

    def test_benchmark_with_random_model(self):
        """Test that benchmark runs with a random model (score near chance)."""
        from brainscore_vision.benchmarks.synthetic_temporal_matching.benchmark import (
            SyntheticTemporalMatching
        )
        from brainscore_vision.model_helpers.brain_transformation.behavior import (
            TemporalMatchToSample
        )

        benchmark = SyntheticTemporalMatching(n_trials=10, n_choices=3, seed=42)

        # Create a model with random features
        activations_model = MockTemporalActivationsModel()
        model = TemporalMatchToSample(
            identifier='mock-random',
            activations_model=activations_model,
            layer=['mock_layer']
        )

        score = benchmark(model)

        # Score should be computed
        assert 'raw' in score.attrs
        assert 'ceiling' in score.attrs
        assert score.attrs['ceiling'] == 1.0

    def test_benchmark_with_perfect_model(self):
        """Test that benchmark returns high score for model that matches directions."""
        from brainscore_vision.benchmarks.synthetic_temporal_matching.benchmark import (
            SyntheticTemporalMatching
        )
        from brainscore_vision.model_helpers.brain_transformation.behavior import (
            TemporalMatchToSample
        )

        benchmark = SyntheticTemporalMatching(n_trials=10, n_choices=3, seed=42)

        # Create a model that encodes direction in features
        activations_model = MockPerfectTemporalModel()
        model = TemporalMatchToSample(
            identifier='mock-perfect',
            activations_model=activations_model,
            layer=['mock_layer']
        )

        score = benchmark(model)

        # Perfect model should get perfect score
        assert score.attrs['raw'] == 1.0
        assert float(score) == 1.0

    def test_benchmark_ceiling(self):
        from brainscore_vision.benchmarks.synthetic_temporal_matching.benchmark import (
            SyntheticTemporalMatching
        )

        benchmark = SyntheticTemporalMatching(n_trials=5, n_choices=3, seed=42)

        # Ceiling should be 1.0 for synthetic data
        assert float(benchmark.ceiling) == 1.0


class TestBenchmarkRegistry:
    def test_benchmark_registered(self):
        """Test that benchmark is properly registered."""
        from brainscore_vision import benchmark_registry

        # Import to trigger registration
        import brainscore_vision.benchmarks.synthetic_temporal_matching

        assert 'SyntheticTemporalMatching' in benchmark_registry
