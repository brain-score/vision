import itertools
from functools import reduce
import operator

import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainscore_core import Metric
from brainscore_vision.metrics import Score
from brainscore_vision.metric_helpers.transformations import apply_aggregate


class AccuracyDistance(Metric):
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
            BehavioralAssembly, variables: tuple = ()) -> Score:
        """Target should be the entire BehavioralAssembly, containing truth values."""

        subjects = self.extract_subjects(target)
        subject_scores = []
        for subject in subjects:
            subject_assembly = target.sel(subject=subject)
            subject_score = self.condition_filtered_score_per_subject_source_pair(source=source,
                                                                                  subject=subject_assembly,
                                                                                  variables=variables)

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

        # .flatten() because models return lists of lists, and here we compare subject-by-subject
        source_correct = source.values.flatten() == target['truth'].values
        target_correct = target.values == target['truth'].values
        source_mean = sum(source_correct) / len(source_correct)
        target_mean = sum(target_correct) / len(target_correct)

        maximum_distance = np.max([1 - target_mean, target_mean])
        # get the proportion of the distance between the source and target accuracies, adjusted for the maximum possible
        # difference between the two accuracies
        relative_distance = 1 - np.abs(source_mean - target_mean) / maximum_distance

        return Score(relative_distance)

    def ceiling(self, assembly, variables = ()):
        subjects = self.extract_subjects(assembly)
        subject_scores = []
        for subject1, subject2 in itertools.combinations(subjects, 2):
            subject1_assembly = assembly.sel(subject=subject1)
            subject2_assembly = assembly.sel(subject=subject2)

            pairwise_score = self.condition_filtered_score_per_subject_source_pair(
                subject1_assembly, subject2_assembly, variables=variables)

            pairwise_score = pairwise_score.expand_dims('subject')
            pairwise_score['subject_left'] = 'subject', [subject1]
            pairwise_score['subject_right'] = 'subject', [subject2]
            subject_scores.append(Score(pairwise_score))

        subject_scores = Score.merge(*subject_scores)
        subject_scores = apply_aggregate(aggregate_fnc=self.aggregate, values=subject_scores)
        return subject_scores

    def extract_subjects(self, assembly):
        return list(sorted(set(assembly['subject'].values)))

    def condition_filtered_score_per_subject_source_pair(self, source, subject, variables):
        # compute single score across the entire dataset
        if len(variables) == 0:
            subject_score = self.compare_single_subject(source, subject)

        # compute scores for each condition, then average
        else:
            cond_scores = []
            # get iterator across all combinations of variables
            if len(variables) == 1:
                conditions = set(subject[variables[0]].values)
                conditions = [[c] for c in conditions]  # to mimic itertools.product
            else:
                # get all combinations of variables that are present in both assemblies
                conditions = itertools.product(
                    *[set(subject[v].values).intersection(set(source[v].values)) for v in variables]
                )

            # loop over conditions and compute scores
            # TODO: 91 conditions?? where do they come from?
            # TODO: what did which participants do? is grouping across visibility correct or not? it does not look
            #  like any participants did the same combination of conditions

            # TODO: but the above is with the caveat that the stimulus_id field is NOT actually the stimulus id.
            #  it contains the subject number and trial number, in addition to a 5-digit number that may or may not
            #  be the actual stimulus_id. we should filter based on that, but need to clean up the data to do that.
            for cond in conditions:
                # filter assemblies for selected condition
                subject_cond_assembly = self.get_condition_filtered_assembly(subject, variables, cond)
                source_cond_assembly = self.get_condition_filtered_assembly(source, variables, cond)
                # to accomodate cases where not all conditions are present in both assemblies, filter out
                #  calculation of the metric for cases where either assembly has no matches to variables (empty)
                if len(subject_cond_assembly) and len(source_cond_assembly):
                    cond_scores.append(self.compare_single_subject(
                        source_cond_assembly, subject_cond_assembly))

            subject_score = Score(np.mean(cond_scores))
        return subject_score

    @staticmethod
    def get_condition_filtered_assembly(assembly, variables, cond):
        # get the indexers for the condition
        indexers = {v: cond[i] for i, v in enumerate(variables)}
        # convert indexers into a list of boolean arrays for the assembly values
        assembly_indexers = [(assembly[key] == value) for key, value in indexers.items()]
        # combine the different conditions into an AND statement to require all conditions simultaneously
        condition = reduce(operator.and_, assembly_indexers)
        # filter the assembly based on the condition
        condition_filtered_assembly = assembly.where(condition, drop=True)
        return condition_filtered_assembly

