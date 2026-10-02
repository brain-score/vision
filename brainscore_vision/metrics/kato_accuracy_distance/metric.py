import itertools

import numpy as np

from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core import Metric
from brainscore_vision.metrics import Score
from brainscore_vision.metric_helpers.transformations import apply_aggregate


class KatoAccuracyDistance(Metric):
    """
    Computes the accuracy distance using the relative distance between the
    source and target accuracies, adjusted for the maximum possible
    difference between the two accuracies. By default, the distance is computed
    from a single accuracy score on the entire BehavioralAssembly. However,
    the distance can also be computed on a condition-wise basis using the
    'variables' argument. The advantage of the condition-wise approach is that
    it can separate two models with identical overall accuracy if one exhibits a
    more target-like pattern of performance across conditions.
    """
    def __call__(self, source: BehavioralAssembly, target:
            BehavioralAssembly, variables: tuple=()) -> Score:
        """Target should be the entire BehavioralAssembly, containing truth values."""

        subjects = self.extract_subjects(target)
        subject_scores = []
        for subject in subjects:
            subject_assembly = target.sel(subject=subject)

            # compute single score across the entire dataset
            if len(variables) == 0:
                subject_score = self.compare_single_subject(source, subject_assembly)

            # compute scores for each condition, then average
            else:
                cond_scores = []

                # get iterator across all combinations of variables
                if len(variables) == 1:
                    conditions = set(subject_assembly[variables[0]].values)
                    conditions = [[c] for c in conditions]  # to mimic itertools.product
                else:
                    conditions = itertools.product(
                        *[set(subject_assembly[v].values) for v in variables])

                # loop over conditions and compute scores
                for cond in conditions:
                    indexers = {v: cond[i] for i, v in enumerate(variables)}
                    subject_cond_assembly = subject_assembly.sel(**indexers)
                    source_cond_assembly = source.sel(**indexers)
                    # to accomodate unbalanced designs, skip combinations of
                    # variables that don't exist in both assemblies
                    if len(subject_cond_assembly) and len(source_cond_assembly):
                        cond_scores.append(self.compare_single_subject(
                            source_cond_assembly, subject_cond_assembly))
                subject_score = Score(np.mean(cond_scores))

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

    def compare_single_subject(self, source: BehavioralAssembly, target: BehavioralAssembly):
        source = source.sortby('stimulus_id')
        target = target.sortby('stimulus_id')
        assert (target['stimulus_id'].values == source['stimulus_id'].values).all()

        # Kato2026: only consider images for which both source and target have a valid response (i.e. non-NaN)
        source_values = np.asarray(source.values).ravel()
        target_values = np.asarray(target.values).ravel()
        source_truth_values = np.asarray(source['truth'].values)
        target_truth_values = np.asarray(target['truth'].values)
        valid = ((source_values != '') & (target_values != '') & (source_values != 'NaN') & (target_values != 'NaN'))

        source_values = source_values[valid]
        target_values = target_values[valid]
        source_truth_values = source_truth_values[valid]
        target_truth_values = target_truth_values[valid]
        
        correct_source = source_values == source_truth_values
        correct_target = target_values == target_truth_values

        source_mean = np.mean(correct_source)
        target_mean = np.mean(correct_target)
        
        maximum_distance = np.max([1 - target_mean, target_mean])
        # get the proportion of the distance between the source and target accuracies, adjusted for the maximum possible
        # difference between the two accuracies
        relative_distance = 1 - np.abs(source_mean - target_mean) / maximum_distance

        return Score(relative_distance)

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
