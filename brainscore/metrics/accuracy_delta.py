import itertools

import numpy as np

from brainscore.metrics import Metric, Score
from brainscore.metrics.ceiling import SplitHalvesConsistencyBaker


class AccuracyDelta(Metric):

    def __call__(self, source, target, image_types, isCeiling=False):

        if isCeiling:
            model_delta = self.get_human_delta(source, image_types)
        else:
            model_delta = self.get_model_delta(source, image_types)
        human_delta = self.get_human_delta(target, image_types)

        # calculate metric score:
        score = max((1 - (np.abs(human_delta - model_delta)) / human_delta), 0)

        # calculate error
        error = 0.005

        return score

    def extract_subjects(self, assembly):
        return list(sorted(set(assembly['subject'].values)))

    def get_human_delta(self, target, image_types):
        # calculate human accuracies for [whole, condition]
        condition_scores_human = []
        for image_type in image_types:
            scores = []
            for subject in self.extract_subjects(target):
                this_target = target.sel(image_type=image_type, subject=subject)
                correct_count = np.count_nonzero(this_target.values)
                accuracy = correct_count / len(this_target)
                scores.append(accuracy)

            overall = np.mean(scores)
            condition_scores_human.append(overall)

        # calculate deltas:
        human_delta = condition_scores_human[0] - condition_scores_human[1]
        return human_delta

    def get_model_delta(self, source, image_types):
        # calculate model accuracies for [whole, condition]
        condition_scores_model = []
        for image_type in image_types:
            # raw network accuracy per category, whole condition
            scores = []
            for category in sorted(set(source['animal'].values)):
                this_source = source.sel(animal=category, image_type=image_type)
                correct_count = (this_source.values == category).sum()
                accuracy = correct_count / len(this_source[0])
                scores.append(accuracy)
            # overall accuracy, averaged over 9 categories
            overall = np.mean(scores)
            condition_scores_model.append(overall)
        model_delta = condition_scores_model[0] - condition_scores_model[1]
        return model_delta

