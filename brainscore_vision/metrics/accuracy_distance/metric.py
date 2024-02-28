import itertools

import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainscore_core import Metric
from brainscore_vision.metrics import Score
from brainscore_vision.metric_helpers.transformations import apply_aggregate


class AccuracyDistance(Metric):
    def __call__(self, source: BehavioralAssembly, assembly: BehavioralAssembly) -> Score:
        """Target should be the entire BehavioralAssembly, containing truth values."""

        subjects = self.extract_subjects(assembly)
        subject_scores = []
        for subject in subjects:
            subject_assembly = assembly.sel(subject=subject)
            subject_score = self.compare_single_subject(source, subject_assembly)
            subject_score = subject_score.expand_dims('subject')
            subject_score['subject'] = 'subject', [subject]
            subject_scores.append(subject_score)

        subject_scores = Score.merge(*subject_scores)
        subject_scores = apply_aggregate(aggregate_fnc=self.aggregate, values=subject_scores)
        return subject_scores

    @classmethod
    def aggregate(cls, scores):
        mean = scores.mean('subject')
        score = abs(mean)  # avoid negative scores just in case
        score.attrs['error'] = scores.std('subject')
        return score

    def compare_single_subject(self, source: BehavioralAssembly, subject_assembly: BehavioralAssembly):
        source = source.sortby('stimulus_id')
        subject_assembly = subject_assembly.sortby('stimulus_id')
        assert (subject_assembly['stimulus_id'].values == source['stimulus_id'].values).all()

        source_correct = source.values == subject_assembly['truth'].values
        target_correct = subject_assembly.values == subject_assembly['truth'].values

        source_is_target = source_correct == target_correct
        accuracy_distance_score = np.mean(source_is_target)
        return Score(accuracy_distance_score)

    def ceiling(self, assembly):

        subjects = self.extract_subjects(assembly)
        subject_scores = []
        for subject1, subject2 in itertools.combinations(subjects, 2):
            subject1_assembly = assembly.sel(subject=subject1)
            subject2_assembly = assembly.sel(subject=subject2)
            pairwise_score = self.compare_single_subject(subject1_assembly, subject2_assembly)
            pairwise_score = pairwise_score.expand_dims('subject')
            pairwise_score['subject_left'] = 'subject', [subject1]
            pairwise_score['subject_right'] = 'subject', [subject2]
            subject_scores.append(Score(pairwise_score))

        subject_scores = Score.merge(*subject_scores)
        subject_scores = apply_aggregate(aggregate_fnc=self.aggregate, values=subject_scores)
        return subject_scores

    def extract_subjects(self, assembly):
        return list(sorted(set(assembly['subject'].values)))