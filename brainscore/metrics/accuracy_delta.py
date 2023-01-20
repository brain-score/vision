import itertools

import numpy as np

from brainscore.metrics import Metric, Score
import random
from brainio.assemblies import BehavioralAssembly


# controls how many half-splits are averaged together to get human delta.
HUMAN_SPLITS = 100


class AccuracyDelta(Metric):

    def __init__(self, image_types: list):
        self.image_types = image_types

    def __call__(self, source: BehavioralAssembly, target: BehavioralAssembly):

        model_delta = get_model_delta(source, self.image_types)
        scores = []

        # calculate score over average of 100 sub splits of human delta
        for i in range(HUMAN_SPLITS):
            human_delta = get_human_delta(target, self.image_types)
            score = max((1 - ((np.abs(human_delta - model_delta)) / human_delta)), 0)
            scores.append(score)
        score = np.mean(scores)
        error = np.std(scores)

        return Score([score, error], coords={'aggregation': ['center', 'error']}, dims=('aggregation',))


def extract_subjects(assembly):
    return list(sorted(set(assembly['subject'].values)))


def get_human_delta(target, image_types, isCeiling=False):
    # calculate human accuracies for [whole, condition]
    condition_scores_human = {}

    if isCeiling:
        half1 = target
    else:
        # pull half of subjects. This is outside loop to make sure
        # the same half of subjects are used in w and f conditions.
        num_subjects = len(set(target["subject"].values))
        half1_subjects = random.sample(range(1, num_subjects), num_subjects // 2)
        half1 = target[{'presentation': [subject in half1_subjects for subject in target['subject'].values]}]

    # for whole condition, and other condition (frank or frag)
    for image_type in image_types:
        scores = []

        # get per subject accuracy
        for subject in extract_subjects(half1):
            this_target = target.sel(image_type=image_type, subject=subject)
            correct_count = np.count_nonzero(this_target.values)
            accuracy = correct_count / len(this_target)
            scores.append(accuracy)
        condition_scores_human[image_type] = scores

    # calculate 16-human pairwise accuracy:
    condition = image_types[1]
    delta_vector = [a_i - b_i for a_i, b_i in zip(condition_scores_human["w"], condition_scores_human[condition])]

    # return mean of delta vector. This is equal to the mean of half of subject deltas (random half)
    return np.mean(delta_vector)


def get_model_delta(source, image_types):
    condition_scores_model = []

    # for whole condition, and other condition (frank or frag)
    for image_type in image_types:

        # raw network accuracy per category
        scores = []
        for category in sorted(set(source['animal'].values)):
            this_source = source.sel(animal=category, image_type=image_type)
            correct_count = (this_source.values == category).sum()
            accuracy = correct_count / len(this_source[0])
            scores.append(accuracy)

        # overall accuracy, averaged over 9 categories
        overall = np.mean(scores)
        condition_scores_model.append(overall)

    # return difference between whole and condition
    model_delta = condition_scores_model[0] - condition_scores_model[1]
    return model_delta


class AccuracyDeltaCeiling:
    def __init__(self, image_types: list):
        self.image_types = image_types

    def __call__(self, source: BehavioralAssembly, target: BehavioralAssembly):
        ceiling = self.compute_ceiling(source, target)
        return ceiling

    def compute_ceiling(self, source, target):
        half_1_delta = get_human_delta(target=source, image_types=self.image_types, isCeiling=True)
        half_2_delta = get_human_delta(target=target, image_types=self.image_types, isCeiling=True)
        score = max((1 - ((np.abs(half_1_delta - half_2_delta)) / half_2_delta)), 0)
        return score
