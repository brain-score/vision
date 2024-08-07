import pandas as pd
import numpy as np
from brainio.assemblies import BehavioralAssembly
import sympy as sp
from pandas import DataFrame
from tqdm import tqdm
import statistics
from typing import Dict

# number of distractors in the experiment
DISTRACTOR_NUMS = ["1.0", "5.0", "11.0"]

# These are precomputed subject average lapse rates
LAPSE_RATES = {'circle_line': 0.0335, 'color': 0.0578, 'convergence': 0.0372, 'eighth': 0.05556,
               'gray_easy': 0.0414, 'gray_hard': 0.02305, 'half': 0.0637, 'juncture': 0.3715,
               'lle': 0.0573, 'llh': 0.0402, 'quarter': 0.0534, 'round_f': 0.08196,
               'round_v': 0.0561, 'tilted_line': 0.04986}

# These are precomputed integral errors, computed by bootstrapping (see below)
HUMAN_INTEGRAL_ERRORS = {'circle_line': 0.3078, 'color': 0.362, 'convergence': 0.2773, 'eighth': 0.278,
                         'gray_easy': 0.309, 'gray_hard': 0.4246, 'half': 0.3661, 'juncture': 0.2198,
                         'lle': 0.209, 'llh': 0.195, 'quarter': 0.2959, 'round_f': 0.344,
                         'round_v': 0.2794, 'tilted_line': 0.3573}


def get_adjusted_rate(acc: float, lapse_rate: float, n_way: int = 2) -> float:
    """
    - Adjusts the raw accuracy by a lapse rate correction

    :param acc: float, the raw accuracy
    :param lapse_rate: a precomputed float defined above that represents avg. subject lapse rate in experiment
    :param n_way: int, (default value 2), the number of ways to divide by

    :return: float, the SEM of that array
    """
    return (acc - lapse_rate * (1.0 / n_way)) / (1 - lapse_rate)


def sem(array: BehavioralAssembly) -> float:
    """
    - Get the standard error of the mean (SEM) of an assembly

    :param array: the assembly to look at
    :return: float, the SEM of that array
    """
    array = np.array(array)
    return np.std(array) / np.sqrt(len(array))


def get_line(point_1: tuple, point_2: tuple) -> str:
    """
    - Calculate the equation of a line from two points

    :param point_1: tuple in the form (x, y) of first point
    :param point_2: tuple in the form (x, y) of second point
    :return: str, equation of a line in the form y = mx + b
    """
    x1, y1 = point_1
    x2, y2 = point_2
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    equation = f"{m:.10f}*x + {c:.10f}"
    return equation


def integrate_line(equation: str, lower: float, upper: float) -> float:
    """
    - Integrates an equation.

    :param equation: a string representing the equation of the line to integrate
    :param lower: float, the lower bound of the definite integral
    :param upper: float, the upper bound of the definite integral
    :return: float, representing the integral of that line
    """
    x = sp.symbols('x')
    integral_definite = sp.integrate(equation, (x, lower, upper))
    return integral_definite


def get_averages(df_blue: DataFrame, df_orange: DataFrame, num_distractors: str) -> (float, float):
    """
    - Gets the per-distractor averages for a block

    :param df_blue: the first (blue) block of data (target on a field of distractors)
    :param df_orange: the second (orange) block of data (distractor on a field of targets)
    :param num_distractors:  string of a float representing how many distractors to look at
    :return:
    """
    blue_df = df_blue[df_blue["distractor_nums"] == num_distractors]
    orange_df = df_orange[df_orange["distractor_nums"] == num_distractors]
    blue_val_avg = blue_df["correct"].values.mean()
    orange_val_avg = orange_df["correct"].values.mean()
    return blue_val_avg, orange_val_avg


def calculate_integral(df_blue: DataFrame, df_orange: DataFrame) -> float:
    """
    - Manually calculates the integral under the delta line

    :param df_blue: the first (blue) block of data (target on a field of distractors)
    :param df_orange: the second (orange) block of data (distractor on a field of targets)
    :return: float representing the integral of the delta line
    """
    blue_low_avg, orange_low_avg = get_averages(df_blue, df_orange, "1.0")
    blue_mid_avg, orange_mid_avg = get_averages(df_blue, df_orange, "5.0")
    blue_high_avg, orange_high_avg = get_averages(df_blue, df_orange, "11.0")

    # compute deltas
    low_delta = orange_low_avg - blue_low_avg
    mid_delta = orange_mid_avg - blue_mid_avg
    high_delta = orange_high_avg - blue_high_avg

    # get equation of line through 1-5
    point_1 = (1, low_delta)
    point_2 = (5, mid_delta)
    equation = get_line(point_1, point_2)
    first_half = integrate_line(equation, 1, 5)

    # get line 5-11 equation and integrate
    point_3 = (11, high_delta)
    equation_2 = get_line(point_2, point_3)
    second_half = integrate_line(equation_2, 5, 11)

    # add up integral
    total_integral = round(first_half + second_half, 4)
    return total_integral


def calculate_accuracy(df: BehavioralAssembly, lapse_rate: float) -> float:
    """
    - Calculates a per-subject lapse rate-corrected accuracy for an assembly.
    - Subject accuracy is averaged over all images with a certain distractor size and repetition coords (i.e. these
      coords are mixed togather and the accuracy is calculated over this merged assembly).

    :param df: DataFrame Object that contains experimental data
    :param lapse_rate: a precomputed float defined above that represents avg. subject lapse rate in experiment
    :return: float representing the adjusted (for lapse rate) accuracy of that subject
    """
    accuracy = len(df[df["correct"] == True]) / len(df)
    adjusted_accuracy = get_adjusted_rate(accuracy, lapse_rate)
    return adjusted_accuracy


def generate_summary_df(assembly: BehavioralAssembly, lapse_rate: float, block: str) -> pd.DataFrame:
    """

    - Takes in raw assembly data and outputs a dataframe of summary statistics, used for benchmark.
    - For each distractor size, accuracy is calculated per subject.

    :param assembly: the data in the form of a BehavioralAssembly
    :param lapse_rate: a precomputed float defined above that represents avg. subject lapse rate in experiment
    :param block: str that defined what data to look at, "first (blue) or second (orange)
    :return: a DataFrame object that contains needed summary data
    """
    filtered_data = assembly[(assembly["trial_type"] == "normal") & (assembly["block"] == block)]
    participants = list(set(filtered_data['participant_id'].values))

    summary_data = []
    for subject in participants:
        subject_data = filtered_data[filtered_data["participant_id"] == subject]
        for distractor_num in DISTRACTOR_NUMS:
            distractor_df = subject_data[subject_data["distractor_nums"] == str(distractor_num)]
            if len(distractor_df) == 0:
                continue
            adjusted_acc = calculate_accuracy(distractor_df, lapse_rate)
            summary_data.append({
                'distractor_nums': distractor_num,
                'participant_id': subject,
                'correct': adjusted_acc
            })
    summary_df = pd.DataFrame(summary_data, columns=['distractor_nums', 'participant_id', 'correct'])
    return summary_df


def split_dataframe(df: BehavioralAssembly, seed: int) -> (BehavioralAssembly, BehavioralAssembly):
    """
    - Takes in one DF and splits it into two, randomly, on the presentation dim

    :param df: The DataFrame (assembly) to split
    :param seed: a seed for the numpy rng
    :return: Two DataFrames (assemblies)
    """
    if seed is not None:
        np.random.seed(seed)
    shuffled_indices = np.random.permutation(df.presentation.size)
    half = len(shuffled_indices) // 2
    indices_1 = shuffled_indices[:half]
    indices_2 = shuffled_indices[half:]
    dataarray_1 = df.isel(presentation=indices_1)
    dataarray_2 = df.isel(presentation=indices_2)
    return dataarray_1, dataarray_2


def get_acc_delta(df_blue: DataFrame, df_orange: DataFrame, num_dist: str) -> float:
    """
    Helper function for bootstrapping. Calculates an accuracy delta on a specific subject/distractor.

    :param df_blue: DataFrame, the first (blue) block of data (target on a field of distractors)
    :param df_orange: DataFrame, the second (orange) block of data (distractor on a field of targets)
    :param num_dist: string, number of distractors
    :return: float representing the requested accuracy delta.
    """
    d_blue = df_blue[df_blue["distractor_nums"] == num_dist]
    d_orange = df_orange[df_orange["distractor_nums"] == num_dist]
    sampled_blue = d_blue.sample(n=1, replace=True)
    sampled_orange = d_orange.sample(n=1, replace=True)
    accuracy_delta = sampled_blue["correct"].values[0] - sampled_orange["correct"].values[0]
    return accuracy_delta


def boostrap_integral(df_blue: DataFrame, df_orange: DataFrame, num_loops: int = 500) -> Dict:
    """
    Computes an error (std) on integral calculation by bootstrapping the integral via slices of subjects.

    :param df_blue: DataFrame, the first (blue) block of data (target on a field of distractors)
    :param df_orange: DataFrame, the second (orange) block of data (distractor on a field of targets)
    :param num_loops: int, number of times the boostrap will run (and thus take the average)
    :return: Dict of values {bootstrapped_integral, bootstrapped_integral_error)
    """
    num_subjects = len(set(df_blue["participant_id"]))
    integral_list = []
    for i in tqdm(range(num_loops)):
        accuracy_delta_lows = []
        accuracy_delta_mids = []
        accuracy_delta_highs = []
        for j in range(num_subjects):
            accuracy_delta_lows.append(get_acc_delta(df_blue, df_orange, num_dist="1.0"))  # get low distractor case
            accuracy_delta_mids.append(get_acc_delta(df_blue, df_orange, num_dist="5.0"))  # get mid distractor case
            accuracy_delta_highs.append(get_acc_delta(df_blue, df_orange, num_dist="11.0"))  # get high distractor case
        average_low_delta = statistics.mean(accuracy_delta_highs)
        average_mid_delta = statistics.mean(accuracy_delta_mids)
        average_high_delta = statistics.mean(accuracy_delta_lows)

        # get equation for line through points 1 - 5 and integrate:
        point_1 = (1, average_low_delta)
        point_2 = (5, average_mid_delta)
        equation = get_line(point_1, point_2)
        first_half = integrate_line(equation, 1, 5)

        # get line 5-11 equation and integrate
        point_3 = (11, average_high_delta)
        equation_2 = get_line(point_2, point_3)
        second_half = integrate_line(equation_2, 5, 11)

        total_integral = first_half + second_half
        integral_list.append(total_integral)
    data_array = np.array(integral_list, dtype=float)
    integral_mean = -np.mean(data_array)
    integral_std = np.std(data_array)

    return {"bootstrap_integral_mean": integral_mean, "integral_std": integral_std}
