import itertools

import numpy as np
from numpy import ndarray

from brainscore.metrics import Metric, Score
from scipy.spatial import distance
from brainio.assemblies import BehavioralAssembly
from typing import Tuple, Dict
import scipy.stats

from brainscore.utils import LazyLoad

NUM_SLOPES = 100  # how many slopes to generate from one round of picking 1 point/cloud at random
NUM_POINTS = 100  # how many data points around the mean to generate
NUM_TRIALS = 500  # how many times we repeat the process

'''
BASIC INFORMATION:

This metric was created in order to score benchmarks on sparse data, or data that might only have key points, such 
as summary data or sparse points on a graph.

The primary use cases are examples where there is only slope (x,y points along with errors) data for humans. 
This metric thus takes those points, along with their errors, and generates more human data via a normal distribution
with mean (data point) and std (from either SEM or STD).

Method steps are below (methods are not in __call__, but are in the form of helper functions:
 1) Generate xx number (default is 3) of clouds of data around the given mean, with a std equal to the given std. 
 2) Pick a point out of each cloud at random, for each cloud. Use these points to calculate a slope, and do this 
    NUM_SLOPES number of times. Return the average slope of these NUM_SLOPES slopes.
 3) Calculate a human index (score) by doing step 2 NUM_TRIALS number of times. The reason to do this multiple times is
    for more samples of average slopes. 
 4) The result then is NUM_TRIALS number of human indexes, each calculated from the average of NUM_SLOPES slopes.


 Ceiling method steps are below:

'''


class DataCloudComparison(Metric):

    def __call__(self, source: float, target):
        # calculate model scores based on each sample human score, to get errors around model score:
        raw_scores = []
        for human_score in target:
            raw_score = max((1 - ((np.abs(human_score - source)) / human_score)), 0)
            # raw_score = max((1 - distance.euclidean(human_score, source)), 0)
            raw_scores.append(raw_score)

        raw_score, model_error = np.mean(raw_scores), np.std(raw_scores)
        raw_score = Score([raw_score, model_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        return raw_score

    def ceiling(self, human_indexes):
        # right now the ceiling simply takes the mean/std of the human indexes. This most likely needs to be changed.
        ceiling, ceiling_error = np.mean(human_indexes), np.std(human_indexes)
        ceiling = Score([ceiling, ceiling_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        return ceiling


# HELPER METHODS BELOW:

# take the means and std, run the trials, and return human indexes
def data_to_indexes(display_sizes, control_means, control_stds, variable_means, variable_stds):
    data_clouds_control = make_new_data(control_means, control_stds)
    data_clouds_variable = make_new_data(variable_means, variable_stds)
    human_indexes = generate_human_data(display_sizes, data_clouds_variable, data_clouds_control)
    return human_indexes


# grab means and bounds (std or assembly default, like SEM) from assembly:
def get_means_stds(discriminator: str, assembly: LazyLoad, use_std=True) -> Tuple[np.array, np.array]:
    shape_trials = assembly.sel(stimulus_id=f'{discriminator}')
    means = shape_trials["response_time"].values

    # use STD or assembly default (could be SEM, etc)
    if use_std:
        num_subjects = set(assembly["num_subjects"].values)
        bounds = (shape_trials["response_time_error"] * np.sqrt(num_subjects.pop())).values
    else:
        bounds = shape_trials["response_time_error"]

    return means, bounds


'''
Generates NUM_POINTS number of possible points in the range of mean +/- std.
Returns a dict of length means with keys of means (int) and values of list (len NUM_POINTS)
'''


def make_new_data(means: np.array, stds: np.array) -> Dict[int, np.array]:
    point_dict = {}
    random_state = np.random.RandomState(0)
    for i in range(len(means)):
        data_points = random_state.normal(means[i], stds[i], size=(1, NUM_POINTS))[0]
        point_dict[means[i]] = data_points
    return point_dict


'''
Takes three clouds of data, picks one point in each cloud, and computes a slope based on those three points.
It does the NUM_POINTS number of times, and returns the average slope of those NUM_POINTS slopes.
Outer/lower limit is there for the random state to ensure fixed trials.
'''


def get_avg_slope(outer_i: int, display_sizes: list, cloud_1: np.array, cloud_2: np.array,
                  cloud_3: np.array) -> ndarray:
    slopes = []
    upper_limit = outer_i * NUM_SLOPES
    lower_limit = upper_limit - NUM_SLOPES
    for i in range(lower_limit, upper_limit):
        random_state = np.random.RandomState(i)
        point_1 = random_state.choice(list(cloud_1))
        point_2 = random_state.choice(list(cloud_2))
        point_3 = random_state.choice(list(cloud_3))
        x_vals = display_sizes
        y_vals = [point_1, point_2, point_3]
        slope_of_best_fit = scipy.stats.linregress(x_vals, y_vals)[0]
        slopes.append(slope_of_best_fit)

    return np.mean(slopes)


# takes in three clouds of data, and generates 500 slopes based on picking a point at random from each cloud.
def generate_human_data(display_sizes: list, shapes_cloud: Dict[int, np.array],
                        cubes_cloud: Dict[int, np.array]) -> list:
    indexes = []
    for i in range(1, NUM_TRIALS + 1):
        d1 = 1 / get_avg_slope(i, display_sizes, list(cubes_cloud.values())[0], list(cubes_cloud.values())[1],
                               list(cubes_cloud.values())[2])
        d2 = 1 / get_avg_slope(i, display_sizes, list(shapes_cloud.values())[0], list(shapes_cloud.values())[1],
                               list(shapes_cloud.values())[2])

        # notched test (replace values)
        # d1 = 1 / get_avg_slope(i, 500, 587, 580)
        # d2 = 1 / get_avg_slope(i, 515, 925, 1085)

        # occluded test (replace values)
        # d1 = 1 / get_avg_slope(i, 576, 656, 709)
        # d2 = 1 / get_avg_slope(i, 517, 708, 828)

        proc_index = (d1 - d2) / (d1 + d2)
        indexes.append(proc_index)
    return indexes