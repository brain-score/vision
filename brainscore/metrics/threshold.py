import itertools
from typing import Dict, Union, Literal

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.special import erf, erfinv

from brainscore.metrics import Metric, Score
from brainscore.metrics.ceiling import SplitHalfConsistency
from brainio.assemblies import PropertyAssembly, BehavioralAssembly, DataAssembly
from brainio.stimuli import StimulusSet


def cumulative_gaussian(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def inverse_cumulative_gaussian(x_point, mu, sigma):
    return np.sqrt(2) * sigma * erfinv(2 * x_point - 1) + mu


class Threshold(Metric):
    """
    Computes a psychometric threshold function from model responses and compares against human-computed psychometric
    thresholds.

    The model comparison to human data is currently individual-subject based, i.e., models and ceilings are compared
    against the mean of the distance of the model threshold to human thresholds.
    """
    def __init__(self,
                 independent_variable,
                 fit_function=cumulative_gaussian,
                 fit_inverse_function=inverse_cumulative_gaussian,
                 threshold_accuracy: Union[Literal['inflection'], float] = 'inflection',
                 scoring: Union[Literal['individual'], Literal['pool']] = 'individual'
                 ):
        self.fit_function = fit_function
        self.fit_inverse_function = fit_inverse_function
        self._independent_variable = independent_variable
        self.threshold_accuracy = threshold_accuracy
        self.scoring = scoring

    def __call__(self, source: Union[np.array, float], target: Union[list, PropertyAssembly]) -> Score:
        """
        :param source: Either a np.array containing model responses to individual stimuli, or a pre-computed threshold
                        as a float.
        :param target: Either a list containing human thresholds (for the ceiling function & ThresholdElevation),
                        or a PropertyAsssembly.
        :return: A Score containing the evaluated model's distance to target thresholds in units of multiples of
                  the human score.
        """
        # compute threshold from measurements if the input is not a threshold already
        if not isinstance(source, float):
            source_threshold = self.compute_threshold(source, self._independent_variable)
        else:
            source_threshold = source

        if source_threshold == 'fit_fail':
            return Score([0.], coords={'aggregation': ['center', ]}, dims=['aggregation'])

        # compare threshold to target thresholds
        if self.scoring == 'pool':
            return self.pool_score(source_threshold, target)
        elif self.scoring == 'individual':
            return self.individual_score(source_threshold, target)
        else:
            raise ValueError(f'Scoring method {self.scoring} is not a valid scoring method.')

    def ceiling(self, assembly: PropertyAssembly):
        """
        :param assembly: the human PropertyAssembly containing human responses
        :return: Score object with coords center (ceiling) and error (STD)
        """
        # compare threshold to target thresholds
        if self.scoring == 'pool':
            return self.pool_ceiling(assembly)
        elif self.scoring == 'individual':
            return self.individual_ceiling(assembly)
        else:
            raise ValueError(f'Scoring method {self.scoring} is not a valid scoring method.')

    def pool_ceiling(self, assembly: PropertyAssembly):
        # Still not super sure what a logical pooled ceiling here is - some split-half procedure like in
        # 'https://github.com/brain-score/brain-score/blob/
        # 9fbf4eda24d081c0ec7bc4d7b5572d8c13dc92d2/brainscore/metrics/image_level_behavior.py#L92'
        # likely makes sense, but is quite problematic with the small amount of target data available in most
        # thresholding studies.
        raise NotImplementedError

    def individual_ceiling(self, assembly: PropertyAssembly):
        """
        Computed by one-vs all for each of the NUM_TRIALS human indexes. One index is removed, and scored against
        a pool of the other values.

        Currently copied with modification from 'https://github.com/brain-score/brain-score/blob/
        jacob2020_occlusion_depth_ordering/brainscore/metrics/data_cloud_comparision.py#L54'.

        :param assembly:
        :return:
        """
        human_thresholds: list = assembly.values.tolist()
        scores = []
        for i in range(len(human_thresholds)):
            random_state = np.random.RandomState(i)
            random_human_score = random_state.choice(human_thresholds, replace=False)
            metric = Threshold(self._independent_variable, self.fit_function, self.fit_inverse_function,
                               self.threshold_accuracy)
            human_thresholds.remove(random_human_score)
            score = metric(random_human_score, human_thresholds)
            score = float(score[(score['aggregation'] == 'center')].values)
            human_thresholds.append(random_human_score)
            scores.append(score)

        ceiling, ceiling_error = np.mean(scores), np.std(scores)
        ceiling = Score([ceiling, ceiling_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        return ceiling

    def compute_threshold(self, source, independent_variable):
        assert len(source.values) == len(source[independent_variable].values)

        x_points = source[independent_variable].values
        accuracies = self.convert_proba_to_correct(source)
        fit_params = self.fit_threshold_function(x_points, accuracies)
        if (type(fit_params) == str) and (fit_params == 'fit_fail'):
            return fit_params

        if self.threshold_accuracy == 'inflection':
            self.threshold_accuracy = self.inflection_accuracy(x_points, fit_params)

        threshold = self.find_threshold(self.threshold_accuracy, fit_params)
        #plot_psychometric_curve(fit_params[0], fit_params[1], scatter=(x_points, accuracies))
        return threshold

    def fit_threshold_function(self, x_points, y_points):
        initial_guess = [np.mean(x_points), np.mean(x_points)]
        try:
            fit = curve_fit(self.fit_function, x_points, y_points, p0=initial_guess)
            # curve_fit returns a ndarray of which the 0th element are the optimized parameters
            params = fit[0].flatten()
            return params
        except RuntimeError:
            print('Model threshold fit unsuccessful. This is likely because of the model outputting the same value '
                  'for every input.')
            return 'fit_fail'

    def find_threshold(self, threshold_accuracy, fit_params):
        threshold = self.fit_inverse_function(threshold_accuracy, *fit_params)
        return threshold

    def inflection_accuracy(self, x_points, fit_params):
        """
        A function that finds the accuracy at the inflection point of the fit function. Useful if you do not care
        about the specific threshold accuracy, but rather about e.g. the elevation at the inflection point.
        """
        max_fit_accuracy = self.fit_function(np.max(x_points), *fit_params)
        min_fit_accuracy = self.fit_function(np.min(x_points), *fit_params)
        threshold_accuracy = min_fit_accuracy + (max_fit_accuracy - min_fit_accuracy) / 2
        return threshold_accuracy

    @staticmethod
    def individual_score(source_threshold, target):
        raw_scores = []
        for human_threshold in target:
            raw_score = max((1 - ((np.abs(human_threshold - source_threshold)) / human_threshold)), 0)
            raw_scores.append(raw_score)

        raw_score, model_error = np.mean(raw_scores), np.std(raw_scores)
        raw_score = Score([raw_score, model_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        return raw_score

    @staticmethod
    def pool_score(source_threshold, target):
        if not isinstance(target, list):
            human_mean = np.mean(target.values)
        else:
            human_mean = np.mean(target)
        raw_score = max((1 - ((np.abs(human_mean - source_threshold)) / human_mean)), 0)
        raw_score = Score([raw_score], coords={'aggregation': ['center', ]}, dims=['aggregation'])
        return raw_score

    @staticmethod
    def convert_proba_to_correct(source):
        decisions = np.argmax(source.values, axis=1)
        correct = []
        for presentation, decision in enumerate(decisions):
            if source['choice'].values[decision] == source['image_label'].values[presentation]:
                correct.append(1)
            else:
                correct.append(0)
        return np.array(correct)

    @staticmethod
    def remove_data_after_asymptote(x):
        """NOTE: CURRENTLY NOT IN USE"""
        # reverse array to get the last occurrence of the max in case of duplicate maxes
        last_max_index = np.argmax(x[::-1])
        return x[:last_max_index]


class ThresholdElevation(Threshold):
    def __init__(self,
                 independent_variable: str,
                 baseline_condition: str,
                 test_condition: str,
                 threshold_accuracy: Union[Literal['inflection'], float] = 'inflection',
                 scoring: Union[Literal['individual'], Literal['pool']] = 'individual'
                 ):
        super(ThresholdElevation, self).__init__(independent_variable)
        self.baseline_threshold_metric = Threshold(self._independent_variable,
                                                   threshold_accuracy=threshold_accuracy)
        self.test_threshold_metric = Threshold(self._independent_variable,
                                               threshold_accuracy=threshold_accuracy)
        self.baseline_condition = baseline_condition
        self.test_condition = test_condition
        self.threshold_accuracy = threshold_accuracy
        self.scoring = scoring

    def __call__(self,
                 source: Union[float, Dict[str, np.array]],
                 target: Union[list, Dict[str, PropertyAssembly]]
                 ) -> Score:
        if isinstance(source, Dict):
            source_baseline_threshold = self.baseline_threshold_metric.compute_threshold(source[self.baseline_condition],
                                                                                         self._independent_variable)
            if self.threshold_accuracy == 'inflection':
                self.test_threshold_metric.threshold_accuracy = self.baseline_threshold_metric.threshold_accuracy
            source_test_threshold = self.test_threshold_metric.compute_threshold(source[self.test_condition],
                                                                                 self._independent_variable)
            raw_source_threshold_elevation = source_test_threshold / source_baseline_threshold
        else:
            raw_source_threshold_elevation = source

        if isinstance(target, Dict):
            target_threshold_elevations = self.compute_threshold_elevations(target)
        else:
            target_threshold_elevations = target

        # compare threshold to target thresholds
        if self.scoring == 'pool':
            return self.pool_score(raw_source_threshold_elevation, target_threshold_elevations)
        elif self.scoring == 'individual':
            return self.individual_score(raw_source_threshold_elevation, target_threshold_elevations)
        else:
            raise ValueError(f'Scoring method {self.scoring} is not a valid scoring method.')

    def ceiling(self, assemblies: Dict[str, PropertyAssembly]):
        if self.scoring == 'pool':
            return self.pool_ceiling(assemblies)
        elif self.scoring == 'individual':
            return self.individual_ceiling(assemblies)
        else:
            raise ValueError(f'Scoring method {self.scoring} is not a valid scoring method.')

    def pool_ceiling(self, assemblies: Dict[str, PropertyAssembly]):
        # Still not super sure what a logical pooled ceiling here is - some split-half procedure like in
        # 'https://github.com/brain-score/brain-score/blob/
        # 9fbf4eda24d081c0ec7bc4d7b5572d8c13dc92d2/brainscore/metrics/image_level_behavior.py#L92'
        # likely makes sense, but is quite problematic with the small amount of target data available in most
        # thresholding studies.
        raise NotImplementedError

    def individual_ceiling(self, assemblies: Dict[str, PropertyAssembly]):
        """
        Computed by one-vs all for each of the NUM_TRIALS human indexes. One index is removed, and scored against
        a pool of the other values.

        Currently copied with modification from 'https://github.com/brain-score/brain-score/blob/
        jacob2020_occlusion_depth_ordering/brainscore/metrics/data_cloud_comparision.py#L54'.

        :param assemblies:
        :return:
        """
        baseline_assembly = assemblies['baseline_assembly']
        condition_assembly = assemblies['condition_assembly']
        human_threshold_elevations = list(condition_assembly.values / baseline_assembly.values)
        scores = []
        for i in range(len(human_threshold_elevations)):
            random_state = np.random.RandomState(i)
            random_human_score = random_state.choice(human_threshold_elevations, replace=False)
            metric = ThresholdElevation(self._independent_variable, self.baseline_condition, self.test_condition,
                                        self.threshold_accuracy, self.scoring)
            human_threshold_elevations.remove(random_human_score)
            score = metric(random_human_score, human_threshold_elevations)
            score = float(score[(score['aggregation'] == 'center')].values)
            human_threshold_elevations.append(random_human_score)
            scores.append(score)

        ceiling, ceiling_error = np.mean(scores), np.std(scores)
        ceiling = Score([ceiling, ceiling_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        return ceiling

    def compute_threshold_elevations(self, assemblies: Dict[str, PropertyAssembly]) -> list:
        baseline_assembly = assemblies['baseline_assembly']
        condition_assembly = assemblies['condition_assembly']
        threshold_elevations = []
        for i, baseline_threshold in baseline_assembly.values:
            condition_threshold = condition_assembly[i]
            threshold_elevations.append(condition_threshold / baseline_threshold)
        return threshold_elevations

