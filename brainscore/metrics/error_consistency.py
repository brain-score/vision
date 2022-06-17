import itertools

import numpy as np

from brainscore.metrics import Metric, Score
from brainscore.metrics.transformations import apply_aggregate


class ErrorConsistency(Metric):
    """
    Computes the error consistency using :meth:`~brainscore.metrics.error_consistency.cohens_kappa`
    implemented in Geirhos et al., 2020 https://arxiv.org/abs/2006.16736
    """

    def __call__(self, source, target):
        assert len(set(source['image_id'].values)) == len(set(target['image_id'].values))
        # from https://github.com/bethgelab/model-vs-human/blob/745046c4d82ff884af618756bd6a5f47b6f36c45/modelvshuman/plotting/analyses.py#L161

        subject_scores = []
        for subject in self.extract_subjects(target):
            for condition in sorted(set(target['condition'].values)):
                this_source = source.sel(condition=condition)
                this_target = target.sel(subject=subject, condition=condition)
                subject_score = self.compare_single_subject(this_source, this_target)
                subject_score = subject_score.expand_dims('subject').expand_dims('condition')
                subject_score['subject'] = [subject]
                subject_score['condition'] = [condition]
                subject_scores.append(subject_score)
        subject_scores = Score.merge(*subject_scores)
        subject_scores = apply_aggregate(aggregate_fnc=self.aggregate, values=subject_scores)
        return subject_scores

    @classmethod
    def aggregate(cls, scores):
        center = scores.mean('condition').mean('subject')
        error = scores.std(['condition', 'subject'])
        return Score([center, error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])

    def ceiling(self, assembly):
        """
        Computes subject consistency by comparing each subject to all other subjects.
        """
        subjects = self.extract_subjects(assembly)
        subject_scores = []
        for subject1, subject2 in itertools.combinations(subjects, 2):
            for condition in sorted(set(assembly['condition'].values)):
                subject1_assembly = assembly.sel(subject=subject1, condition=condition)
                subject2_assembly = assembly.sel(subject=subject2, condition=condition)
                pairwise_score = self.compare_single_subject(subject1_assembly, subject2_assembly)
                pairwise_score = pairwise_score.expand_dims('subject').expand_dims('condition')
                pairwise_score['subject_left'] = 'subject', [subject1]
                pairwise_score['subject_right'] = 'subject', [subject2]
                pairwise_score['condition'] = [condition]
                subject_scores.append(Score(pairwise_score))
        subject_scores = Score.merge(*subject_scores)
        subject_scores = apply_aggregate(aggregate_fnc=lambda scores: subject_scores.mean('condition').mean('subject'),
                                         values=subject_scores)
        return subject_scores

    def extract_subjects(self, assembly):
        return list(sorted(set(assembly['subject'].values)))

    def compare_single_subject(self, source, target):
        assert len(source['presentation']) == len(target['presentation'])
        source = source.sortby('image_id')
        target = target.sortby('image_id')
        assert all(source['image_id'].values == target['image_id'].values)

        correct_source = source.values == source['truth'].values
        correct_target = target.values == target['truth'].values
        accuracy_source = np.mean(correct_source)
        accuracy_target = np.mean(correct_target)

        expected_consistency = accuracy_source * accuracy_target + (1 - accuracy_source) * (1 - accuracy_target)
        observed_consistency = (correct_source == correct_target).sum() / len(target)
        error_consistency = cohens_kappa(expected_consistency=expected_consistency,
                                         observed_consistency=observed_consistency)
        return Score(error_consistency)


def cohens_kappa(expected_consistency, observed_consistency):
    """
    Computes the error consistency using Cohen's Kappa.
    Cohen, 1960 https://doi.org/10.1177%2F001316446002000104
    """
    # from https://github.com/bethgelab/model-vs-human/blob/745046c4d82ff884af618756bd6a5f47b6f36c45/modelvshuman/plotting/analyses.py#L147-L158
    assert 0.0 <= expected_consistency <= 1.0
    assert 0.0 <= observed_consistency <= 1.0
    return (observed_consistency - expected_consistency) / (1.0 - expected_consistency)
