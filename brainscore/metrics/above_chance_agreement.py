import itertools

import numpy as np
import scipy.stats
import xarray as xr

from brainscore.metrics import Metric, Score
from brainscore.metrics.transformations import apply_aggregate


class AboveChanceAgreement(Metric):

    def __call__(self, source, target, image_type):
        assert len(set(source['stimulus_id'].values)) == len(set(target['stimulus_id'].values))
        # https://github.com/bethgelab/model-vs-human/blob/745046c4d82ff884af618756bd6a5f47b6f36c45/modelvshuman/plotting/analyses.py#L161
        subject_scores = []
        for subject in self.extract_subjects(target):
            for category in sorted(set(target['animal'].values)):
                this_target = target.sel(subject=1, animal=category, image_type=image_type)
                source_images = this_target["stimulus_id"].values
                mask = source["stimulus_id"].isin(source_images)
                this_source = source.where(mask).dropna("presentation").sel(image_type=image_type)
                subject_score = self.compare_single_subject(this_source, this_target)
                subject_score = subject_score.expand_dims('subject').expand_dims('category')
                subject_score['subject'] = [subject]
                subject_score['category'] = [category]
                subject_scores.append(subject_score)
        subject_scores = Score.merge(*subject_scores)
        subject_scores = apply_aggregate(aggregate_fnc=self.aggregate, values=subject_scores)
        return subject_scores

    @classmethod
    def aggregate(cls, scores):
        center = scores.mean('category').mean('subject')
        error = scores.std(['category', 'subject'])  # note that the original paper did not have error estimates
        # This deviates from the original paper which did not deal with scores < 0
        # (neither for a single subject/condition, nor in the aggregate).
        # We here avoid negative scores, so that we comply with all 0 <= scores <= 1.
        center = np.abs(center)
        return Score([center, error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])

    def ceiling(self, assembly, image_type):
        """
        Computes subject consistency by comparing each subject to all other subjects.
        """
        subjects = sorted(set(assembly['subject'].values))
        subject_scores = []
        for subject1 in subjects:
            for category in sorted(set(assembly['animal'].values)):
                # get correct images
                subject1_assembly = assembly.sel(subject=subject1, animal=category, image_type=image_type)
                sub1_images = list(set(subject1_assembly["stimulus_id"].values))

                pool_mask = assembly["stimulus_id"].isin(sub1_images)
                pool_trials = assembly.where(pool_mask).dropna("presentation")
                pool_trials = pool_trials.where(pool_trials['subject'] != subject1).dropna('presentation')

                # get the subject correct vector
                sub1 = subject1_assembly["correct"].values
                subject_pool = set(pool_trials["subject"].values)

                assert set(pool_trials["stimulus_id"].values) == set(sub1_images)

                print(f"comparing  subject {subject1} to subject pool{subject_pool}")

                # get the mode vector over subjects who saw same images:
                modes = []
                for image in subject1_assembly["stimulus_id"].values:
                    mode = scipy.stats.mode(pool_trials.sel(stimulus_id=image)["correct"].values)[0][0]
                    modes.append(mode)

                pairwise_score = self.compare_single_subject(sub1, np.array(modes), is_ceiling=True)
                pairwise_score = pairwise_score.expand_dims('subject').expand_dims("category")
                pairwise_score['subject_left'] = 'subject', [subject1]
                pairwise_score['subject_pool'] = 'subject', [str(subject_pool)]
                pairwise_score['category'] = [category]
                subject_scores.append(Score(pairwise_score))
        subject_scores = Score.merge(*subject_scores)
        subject_scores = apply_aggregate(aggregate_fnc=self.aggregate, values=subject_scores)
        return subject_scores

    def extract_subjects(self, assembly):
        return list(sorted(set(assembly['subject'].values)))

    def compare_single_subject(self, source, target, is_ceiling=False):

        if is_ceiling:
            correct_source = source
            correct_target = target
        else:
            assert len(source['presentation']) == len(target['presentation'])
            source = source.sortby('stimulus_id')
            target = target.sortby('stimulus_id')
            assert all(source['stimulus_id'].values == target['stimulus_id'].values)
            correct_source = source.values == source['animal'].values
            correct_target = target['correct'].values

        accuracy_source = np.mean(correct_source)
        accuracy_target = np.mean(correct_target)
        expected_consistency = accuracy_source * accuracy_target + (1 - accuracy_source) * (1 - accuracy_target)
        observed_consistency = (correct_source == correct_target).sum() / len(target)
        aca = observed_consistency - expected_consistency
        return Score(aca)
