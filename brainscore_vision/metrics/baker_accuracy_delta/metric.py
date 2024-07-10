from typing import List

import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainscore_core import Metric, Score

# controls how many half-splits are averaged together to get human delta.
HUMAN_SPLITS = 100


class BakerAccuracyDelta(Metric):
    def __init__(self, image_types: list):
        self.image_types = image_types

    def __call__(self, source: BehavioralAssembly, target: BehavioralAssembly):
        model_delta = get_model_delta(source, self.image_types)
        scores = []

        # calculate score over average of 100 sub splits of human delta
        for i in range(HUMAN_SPLITS):
            # grab one half of the subjeects
            random_state = np.random.RandomState(i)
            num_subjects = len(set(target["subject"].values))
            half1_subjects = random_state.choice(range(1, num_subjects), (num_subjects // 2), replace=False)
            half1 = target[{'presentation': [subject in half1_subjects for subject in target['subject'].values]}]

            human_delta = get_human_delta(half1, self.image_types)
            score = max((1 - ((np.abs(human_delta - model_delta)) / human_delta)), 0)
            scores.append(score)
        score = np.mean(scores)
        error = np.std(scores)

        score = Score(score)
        score.attrs['error'] = error
        score.attrs['raw'] = scores
        return score

    def compute_ceiling(self, source: BehavioralAssembly, target: BehavioralAssembly, image_types: List[str]) -> float:
        half_1_delta = get_human_delta(target=source, image_types=image_types)
        half_2_delta = get_human_delta(target=target, image_types=image_types)
        ceiling = max((1 - ((np.abs(half_1_delta - half_2_delta)) / half_2_delta)), 0)
        return ceiling


def extract_subjects(assembly):
    return list(sorted(set(assembly['subject'].values)))


def get_human_delta(target, image_types):
    # calculate human accuracies for [whole, condition]
    condition_scores_human = {}

    # for whole condition, and other condition (frankenstein or fragmented)
    for image_type in image_types:
        scores = []

        # get per subject accuracy
        for subject in extract_subjects(target):
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
