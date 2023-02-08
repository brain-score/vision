import itertools

import numpy as np

from brainscore.metrics import Metric, Score
import random
from brainio.assemblies import BehavioralAssembly
import random
import scipy.stats
NUM_LINES = 50


class DataCloudComparison(Metric):
    def __init__(self, shape: str):
        self.shape = shape

    def __call__(self, source: float, target):

        clouds = []
        for i in range(6):
            mean, std = get_mean_std(target)
            cloud = make_new_data(mean, std)
            clouds.append(cloud)

        human_indexes = generate_human_data(clouds)

        # calculate model scores based on each sample human score, to get errors around model score:
        raw_scores = []
        for human_score in human_indexes:
            raw_score = max((1 - ((np.abs(human_score - source)) / human_score)), 0)
            raw_scores.append(raw_score)

        raw_score, model_error = np.mean(raw_scores), np.std(raw_scores)
        raw_score = Score([raw_score, model_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])

        ceiling, ceiling_error = np.mean(human_indexes), np.std(human_indexes)
        ceiling = Score([ceiling, ceiling_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])

        return raw_score, ceiling


def make_new_data(mean, std):
    data_points = np.random.normal(mean, std, size=(1, NUM_LINES))
    return data_points


def get_avg_slope(cloud_1, cloud_2, cloud_3):
    slopes = []
    for i in range(NUM_LINES):
        point_1 = random.choice(list(cloud_1)[0])
        point_2 = random.choice(list(cloud_2)[0])
        point_3 = random.choice(list(cloud_3)[0])
        x_vals = [1, 6, 12]
        y_vals = [point_1, point_2, point_3]
        slope_of_best_fit = scipy.stats.linregress(x_vals, y_vals)[0]
        slopes.append(slope_of_best_fit)

    avg_slope = np.mean(slopes)
    return avg_slope


def generate_human_data(cloud_list):
    NUM_TRIALS = 250
    proc_indexes = []
    for i in range(NUM_TRIALS):
        print(i)
        d1 = 1 / get_avg_slope(cloud_list[0], cloud_list[1], cloud_list[2])
        d2 = 1 / get_avg_slope(cloud_list[3], cloud_list[4], cloud_list[5])
        proc_index = (d1 - d2) / (d1 + d2)
        proc_indexes.append(proc_index)
    return proc_indexes