import numpy as np
from numpy import ndarray
from brainscore.metrics import Metric, Score
from scipy.stats import pearsonr
from brainio.assemblies import BehavioralAssembly
from typing import Tuple, Dict
import scipy.stats
from brainscore.utils import LazyLoad

NUM_SLOPES = 55  # how many slopes to generate from one round of picking 1 point/cloud at random
NUM_POINTS = 55  # how many data points around the mean to generate
NUM_TRIALS = 250  # how many times we repeat the process

"""
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

 NOTES:
 1) The values for NUM_SLOPES, NUM_POINTS, NUM_TRIALS were chosen above after trying many combinations. These values 
    seem to give a very close approximation to the original target. 

"""


class DataCloudComparison(Metric):

    def __call__(self, source: float, target: BehavioralAssembly) -> Score:

        # calculate model scores based on each sample human score, to get errors around model score:
        raw_scores = []
        for human_score in target:
            raw_score = max((1 - ((np.abs(human_score - source)) / human_score)), 0)
            # raw_score = max((1 - distance.euclidean(human_score, source)), 0)
            raw_scores.append(raw_score)

        raw_score, model_error = np.mean(raw_scores), np.std(raw_scores)
        raw_score = Score([raw_score, model_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        return raw_score

    def ceiling(self, human_indexes: list) -> Score:
        """
        Computed by one-vs all for each of the NUM_TRIALS human indexes. One index is removed, and scored against
        a pool of the other values.

        :param human_indexes: a list of human 3DPI indexes
        :return: Score object with coords center (ceiling) and error (STD)
        """
        scores = []
        for i in range(len(human_indexes)):
            random_state = np.random.RandomState(i)
            random_human_score = random_state.choice(human_indexes, replace=False)
            metric = DataCloudComparison()
            human_indexes.remove(random_human_score)
            score = metric(random_human_score, human_indexes)
            score = float(score[(score['aggregation'] == 'center')].values)
            human_indexes.append(random_human_score)
            scores.append(score)

        ceiling, ceiling_error = np.mean(scores), np.std(scores)
        ceiling = Score([ceiling, ceiling_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        return ceiling


#
def data_to_indexes(display_sizes: list, control_means: list, control_stds: list,
                    variable_means: list, variable_stds: list, use_alt_scoring=False) -> list:
    """
    Take the means and std, run the trials (generate data), and return human indexes.

    :param use_alt_scoring: If occlusion/depth ordering benchmark, we need to use a different formula.
    :param display_sizes: a list of values to use for the x_axis slope computation. Always will be len 3.
    :param control_means: the means of the image used to compare against.
    :param control_stds: the stds of the image used to compare against.
    :param variable_means: the means of the image of interest.
    :param variable_stds: the stds of the image of interest.
    :return: a list of human indexes, NUM_TRIALS long.
    """
    data_clouds_control = make_new_data(control_means, control_stds)
    data_clouds_variable = make_new_data(variable_means, variable_stds)
    human_indexes = generate_human_data(display_sizes, data_clouds_variable, data_clouds_control, use_alt_scoring)
    return human_indexes


#
def get_means_stds(discriminator: str, assembly: LazyLoad, use_std=True) -> Tuple[np.array, np.array]:
    """
    Helper function to grab means and bounds  from assembly:

    :param discriminator: string controlling sub-benchmarks
    :param assembly: The human data, in the form of a LazyLoad Assembly
    :param use_std: bool, default True that controls whether to use STD or SEM for error.
    :return: Tuple of the form [means, bounds]
    """

    shape_trials = assembly.sel(stimulus_id=f'{discriminator}')
    means = shape_trials["response_time"].values

    # use STD or assembly default (could be SEM, etc)
    if use_std:
        num_subjects = set(assembly["num_subjects"].values)
        bounds = (shape_trials["response_time_error"] * np.sqrt(num_subjects.pop())).values
    else:
        bounds = shape_trials["response_time_error"]

    return means, bounds


def make_new_data(means: np.array, stds: np.array) -> Dict[int, np.array]:
    """
    Generates NUM_POINTS number of possible points in the range of mean +/- std.

    :param means: np array of means (centers)
    :param stds: np array of stds (errors)
    :return: Dict[int, np.array] with key being each mean and value being a np array of len NUM_POINTS
    """
    point_dict = {}
    random_state = np.random.RandomState(0)
    for i in range(len(means)):
        data_points = random_state.normal(means[i], stds[i], size=(1, NUM_POINTS))[0]
        point_dict[means[i]] = data_points
    return point_dict


def get_avg_slope(outer_i: int, display_sizes: list, cloud_1: np.array,
                  cloud_2: np.array, cloud_3: np.array) -> ndarray:
    """
    Computes a slope based on three points. One point is picked at random from each cloud.
    It does the NUM_POINTS number of times and returns the average.

    :param outer_i: Used to control random seed by generating a range with lower_bound and upper_bound,
                    NUM_SLOPES wide. Ensures that re-running will not change outcome.
    :param display_sizes: a list of values to use for the x_axis slope computation. Always will be len 3.
    :param cloud_1: Data Cloud around first mean with STD
    :param cloud_2: Data Cloud around second mean with STD
    :param cloud_3: Data Cloud around third mean with STD
    :return: np array, with a single value equal to the average slope of those NUM_POINTS slopes.
    """
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


def generate_human_data(display_sizes: list, shapes_cloud: Dict[int, np.array],
                        cubes_cloud: Dict[int, np.array], use_alt_scoring) -> list:
    """
    Generates 500 slopes based on picking a point at random from each cloud. Calls get_avg_slope NUM_TRIALS times.

    :param use_alt_scoring: If occlusion/depth ordering benchmark, we need to use a different formula.
    :param display_sizes: a list of values to use for the x_axis slope computation. Always will be len 3.
    :param shapes_cloud: A dict with all means (keys) and their corresponding clouds (values) for shapes
    :param cubes_cloud: A dict with all means (keys) and their corresponding clouds (values) for cubes
    :return: list of indexes for the model, using formula below and based off of Jacob 2020
    """
    indexes = []
    for i in range(1, NUM_TRIALS + 1):
        d1 = 1 / get_avg_slope(i, display_sizes, list(cubes_cloud.values())[0], list(cubes_cloud.values())[1],
                               list(cubes_cloud.values())[2])
        d2 = 1 / get_avg_slope(i, display_sizes, list(shapes_cloud.values())[0], list(shapes_cloud.values())[1],
                               list(shapes_cloud.values())[2])

        # for some reason, in the occlusion/depth ordering experiment, Jacob uses a different scoring method.
        if use_alt_scoring:
            proc_index = (d2 - d1) / (d1 + d2)
        else:
            proc_index = (d1 - d2) / (d1 + d2)
        indexes.append(proc_index)
    return indexes


