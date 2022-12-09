import itertools

import numpy as np
import scipy.stats
import xarray as xr

from brainscore.metrics import Metric, Score
from brainscore.metrics.transformations import apply_aggregate


class AboveChanceAgreement(Metric):

    def __call__(self, source, target):
        assert len(set(source['stimulus_id'].values)) == len(set(target['stimulus_id'].values))
        # https://github.com/bethgelab/model-vs-human/blob/745046c4d82ff884af618756bd6a5f47b6f36c45/modelvshuman/plotting/analyses.py#L161
        subject_scores = []
        image_type = "w"
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

    def ceiling(self, assembly):
        """
        Computes subject consistency by comparing each subject to all other subjects.
        """
        subjects = [1, 2, 3, 4, 5, 6, 7]
        subject_scores = []
        image_type = "w"
        for subject in subjects:
            for category in sorted(set(assembly['animal'].values)):
                single_subject = assembly.sel(subject=subject, animal=category, image_type=image_type)
                all_subjects = assembly.sel(animal=category, image_type=image_type)

                single_subject_images = single_subject["stimulus_id"].values
                mask = all_subjects["stimulus_id"].isin(single_subject_images)
                all_subjects = all_subjects.where(mask).dropna("presentation")

                # get mode of subject responses by image:
                all_subjects = all_subjects.groupby("stimulus_id").mean()

                # test = []
                # for image in single_subject_images:
                #     mode = scipy.stats.mode(all_subjects[all_subjects['stimulus_id'] == image])[0][0]
                #     test.append(mode)


                subject_score = self.compare_single_subject(single_subject, all_subjects)
                subject_score = subject_score.expand_dims('subject').expand_dims('category')
                subject_score['subject'] = [subject]
                subject_score['category'] = [category]
                subject_scores.append(subject_score)
        subject_scores = Score.merge(*subject_scores)
        subject_scores = apply_aggregate(aggregate_fnc=self.aggregate, values=subject_scores)
        return subject_scores
        # for subject1, subject2 in itertools.combinations(subjects, 2):
        #     for category in sorted(set(assembly['nimal'].values)):
        #         subject1_assembly = assembly.sel(subject=subject1, animal=category)
        #         subject2_assembly = assembly.sel(subject=subject2, animal=category)
        #         pairwise_score = self.compare_single_subject(subject1_assembly, subject2_assembly)
        #         pairwise_score = pairwise_score.expand_dims('subject').expand_dims('category')
        #         pairwise_score['subject_left'] = 'subject', [subject1]
        #         pairwise_score['subject_right'] = 'subject', [subject2]
        #         pairwise_score['category'] = [category]
        #         subject_scores.append(Score(pairwise_score))
        # subject_scores = Score.merge(*subject_scores)
        # subject_scores = apply_aggregate(aggregate_fnc=self.aggregate, values=subject_scores)
        # return subject_scores

    def extract_subjects(self, assembly):
        return list(sorted(set(assembly['subject'].values)))

    def compare_single_subject(self, source, target):
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

def get_mode(*args, **kwargs):
    vals = scipy.stats.mode(*args, **kwargs)
    # only return the mode (discard the count)
    return vals[0][0]

def mode(obj, dim):
    final = xr.apply_ufunc(get_mode, obj, input_core_dims=[[dim]])
    return final