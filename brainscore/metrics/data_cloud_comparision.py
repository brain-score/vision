import itertools

import numpy as np

from brainscore.metrics import Metric, Score
from scipy.spatial import distance
import scipy.stats
NUM_LINES = 500
NUM_POINTS = 100


class DataCloudComparison(Metric):
    def __init__(self, shape: str):
        self.shape = shape

    def __call__(self, source: float, target):

        # use cube 2 if shape is y. Cube 2's slope value is 12, which is what is needed
        if self.shape == "y":
            cube_means, cube_stds = get_means_stds("cube_2", target)
        else:
            cube_means, cube_stds = get_means_stds("cube_1", target)

        shape_means, shape_stds = get_means_stds(f"{self.shape}_1", target)
        data_clouds_cube = make_new_data(cube_means, cube_stds)
        data_clouds_shape = make_new_data(shape_means, shape_stds)

        human_indexes = generate_human_data(data_clouds_cube, data_clouds_shape)

        # calculate model scores based on each sample human score, to get errors around model score:
        raw_scores = []
        for human_score in human_indexes:
            #raw_score = max((1 - ((np.abs(human_score - source)) / human_score)), 0)
            raw_score = max((1 - distance.euclidean(human_score, source)), 0)
            raw_scores.append(raw_score)

        raw_score, model_error = np.mean(raw_scores), np.std(raw_scores)
        raw_score = Score([raw_score, model_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])

        ceiling, ceiling_error = np.mean(human_indexes), np.std(human_indexes)
        ceiling = Score([ceiling, ceiling_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])

        return raw_score, ceiling


# grab means and standard deviations from assembly:
def get_means_stds(shape, assembly):

    shape_trials = assembly.sel(stimulus_id=f'{shape}')
    means = shape_trials["response_time"].values

    # calculate standard deviations from standard error means
    num_subjects = set(assembly["num_subjects"].values)
    stds = (shape_trials["response_time_error"] * np.sqrt(num_subjects.pop())).values

    return means, stds


# generates NUM_POINTS number of possible points in the range of mean +/- std
def make_new_data(means, stds):
    points = []
    random_state = np.random.RandomState(0)
    for i in range(len(means)):
        data_points = random_state.normal(means[i], stds[i], size=(1, NUM_POINTS))[0]
        points.append(data_points)
    return points


def get_avg_slope(outer_i, cloud_1, cloud_2, cloud_3):
    slopes = []
    upper_limit = outer_i * NUM_POINTS
    lower_limit = upper_limit - NUM_POINTS
    for i in range(lower_limit, upper_limit):
        random_state = np.random.RandomState(i)
        point_1 = random_state.choice(list(cloud_1))
        point_2 = random_state.choice(list(cloud_2))
        point_3 = random_state.choice(list(cloud_3))
        x_vals = [1, 6, 12]
        y_vals = [point_1, point_2, point_3]
        slope_of_best_fit = scipy.stats.linregress(x_vals, y_vals)[0]
        slopes.append(slope_of_best_fit)

    return np.mean(slopes)


def generate_human_data(cubes_cloud, shapes_cloud):
    indexes = []
    for i in range(1, NUM_LINES + 1):
        d1 = 1 / get_avg_slope(i, cubes_cloud[0], cubes_cloud[1], cubes_cloud[2])
        d2 = 1 / get_avg_slope(i, shapes_cloud[0], shapes_cloud[1], shapes_cloud[2])

        # square test
        # d1 = 1 / get_avg_slope(i, 500, 587, 580)
        # d2 = 1 / get_avg_slope(i, 515, 925, 1085)

        # y test
        # d1 = 1 / get_avg_slope(i, 576, 656, 709)
        # d2 = 1 / get_avg_slope(i, 517, 708, 828)

        proc_index = (d1 - d2) / (d1 + d2)
        indexes.append(proc_index)
    return indexes


