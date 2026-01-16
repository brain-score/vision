"""
Synthetic Temporal Match-to-Sample Benchmark

A synthetic benchmark for validating the temporal match-to-sample infrastructure.
Uses procedurally generated moving dot stimuli where the matching criterion is
motion direction.

This benchmark is primarily for infrastructure testing and validation, not for
production model evaluation.
"""

import numpy as np

from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics import Score
from brainscore_vision.model_interface import BrainModel
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from .stimuli import generate_synthetic_benchmark_stimuli


BIBTEX = """@misc{synthetic_temporal_matching,
    title={Synthetic Temporal Match-to-Sample Benchmark},
    author={Brain-Score Team},
    year={2026},
    note={Infrastructure validation benchmark with synthetic stimuli}
}"""


class SyntheticTemporalMatching(BenchmarkBase):
    """
    Synthetic benchmark for temporal match-to-sample task.

    Generates moving dot stimuli where the sample and one choice share the same
    motion direction. The model must identify which choice matches the sample.

    Ground truth is deterministic (correct choice is known), so ceiling = 1.0.
    This benchmark is for infrastructure validation, not production use.
    """

    def __init__(
        self,
        n_trials: int = 10,
        n_choices: int = 3,
        seed: int = 42
    ):
        """
        :param n_trials: Number of trials in the benchmark
        :param n_choices: Number of choices per trial
        :param seed: Random seed for stimulus generation
        """
        self._n_trials = n_trials
        self._n_choices = n_choices
        self._seed = seed
        self._visual_degrees = 8
        self._number_of_trials = 1

        # Generate stimuli and ground truth
        self._stimuli_data, self._correct_choices = generate_synthetic_benchmark_stimuli(
            n_trials=n_trials,
            n_choices=n_choices,
            seed=seed
        )

        # Build StimulusSet
        self._stimulus_set = self._build_stimulus_set()

        super().__init__(
            identifier='SyntheticTemporalMatching',
            version=1,
            ceiling_func=lambda: Score(1.0),  # Perfect ceiling for synthetic data
            parent='behavior_vision',
            bibtex=BIBTEX
        )

    def _build_stimulus_set(self) -> StimulusSet:
        """Build a StimulusSet from the generated stimuli data."""
        stimulus_ids = [s['stimulus_id'] for s in self._stimuli_data]
        trial_ids = [s['trial_id'] for s in self._stimuli_data]
        stimulus_roles = [s['stimulus_role'] for s in self._stimuli_data]
        choice_indices = [s['choice_index'] for s in self._stimuli_data]
        directions = [s['direction'] for s in self._stimuli_data]
        paths = [s['stimulus_path'] for s in self._stimuli_data]

        stimuli = StimulusSet({
            'stimulus_id': stimulus_ids,
            'trial_id': trial_ids,
            'stimulus_role': stimulus_roles,
            'choice_index': choice_indices,
            'direction': directions,
            'filename': [p.split('/')[-1] for p in paths],
        })

        # Set stimulus paths
        stimuli.stimulus_paths = {
            s['stimulus_id']: s['stimulus_path'] for s in self._stimuli_data
        }
        stimuli.identifier = 'SyntheticTemporalMatching.stimuli'

        return stimuli

    def __call__(self, candidate: BrainModel) -> Score:
        """
        Evaluate a model on the synthetic temporal matching benchmark.

        :param candidate: A BrainModel to evaluate
        :return: Score representing model-ground truth agreement
        """
        # Start the task
        candidate.start_task(BrainModel.Task.match_to_sample)

        # Run the model on stimuli
        choices = candidate.look_at(self._stimulus_set, self._number_of_trials)

        # Score: compare model choices to ground truth
        model_choices = choices.values.flatten()
        correct_choices = np.array(self._correct_choices)

        # Compute accuracy
        correct = model_choices == correct_choices
        raw_score = np.sum(correct) / len(correct)

        # Chance level for n_choices
        chance = 1.0 / self._n_choices

        # Normalize above chance, ceiling is 1.0
        normalized_score = (raw_score - chance) / (1.0 - chance)
        normalized_score = max(0.0, normalized_score)  # Floor at 0

        score = Score(normalized_score)
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = self.ceiling
        score.attrs['n_trials'] = self._n_trials
        score.attrs['n_choices'] = self._n_choices
        score.attrs['chance'] = chance

        return score
