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


def cumulative_gaussian(x: np.array, mu: float, sigma: float) -> float:
    """The cumulative gaussian function."""
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def inverse_cumulative_gaussian(x_point: float, mu: float, sigma: float) -> float:
    """Inverts the cumulative_gaussian function."""
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

    def __call__(self, source: Union[BehavioralAssembly, float], target: Union[list, PropertyAssembly]) -> Score:
        """
        :param source: Either a BehavioralAssembly containing model responses to individual stimuli, or a pre-computed
                        threshold as a float.
        :param target: Either a list containing human thresholds (for the ceiling function & ThresholdElevation),
                        or a PropertyAsssembly.
        :return: A Score containing the evaluated model's ceiling-adjusted distance to target thresholds.
        """
        # compute threshold from measurements if the input is not a threshold already
        if isinstance(source, float):
            source_threshold = source
        elif isinstance(source, BehavioralAssembly):
            source_threshold = self.compute_threshold(source, self._independent_variable)
            # check whether the psychometric function fit was successful - if not, return a score of 0
            if source_threshold == 'fit_fail':
                return Score([0.], coords={'aggregation': ['center', ]}, dims=['aggregation'])
        else:
            raise TypeError(f'source is type {type(source)}, but type BehavioralAssembly or float is required.')

        # compare threshold to target thresholds
        if self.scoring == 'pool':
            return self.pool_score(source_threshold, target)
        elif self.scoring == 'individual':
            return self.individual_score(source_threshold, target)
        else:
            raise ValueError(f'Scoring method {self.scoring} is not a valid scoring method.')

    def ceiling(self, assembly: Union[PropertyAssembly, Dict[str, PropertyAssembly]]) -> Score:
        """
        Selects the appropriate ceiling to be computed from target assembly data.

        :param assembly: the human PropertyAssembly containing human responses, or a dict containing the
                          PropertyAssemblies of the ThresholdElevation metric.
        :return: Score object with coords center (ceiling) and error (STD)
        """
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

    def compute_threshold(self, source: BehavioralAssembly, independent_variable: str) -> float:
        """Converts the source BehavioralAssembly to a threshold float value."""
        assert len(source.values) == len(source[independent_variable].values)

        x_points = source[independent_variable].values
        accuracies = self.convert_proba_to_correct(source)
        fit_params = self.fit_threshold_function(x_points, accuracies)
        if (type(fit_params) == str) and (fit_params == 'fit_fail'):
            return fit_params

        if self.threshold_accuracy == 'inflection':
            self.threshold_accuracy = self.inflection_accuracy(x_points, fit_params)

        threshold = self.find_threshold(self.threshold_accuracy, fit_params)
        return threshold

    def fit_threshold_function(self, x_points: np.array, y_points: np.array) -> Union[np.array, str]:
        """
        A function that takes the x and y-points of the measured variable and handles the fitting of the
        psychometric threshold function.

        Returns either the fit parameters for self.fit_function or a string tag that indicates the failure
        of the psychometric curve fit.
        """
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

    def find_threshold(self, threshold_accuracy: float, fit_params: Tuple[float, ...]) -> float:
        """
        A function that uses the inverse fit function to find the value of the threshold in terms of
        the independent variable (self._independent_variable).
        """
        threshold = self.fit_inverse_function(threshold_accuracy, *fit_params)
        return threshold

    def inflection_accuracy(self, x_points: np.array, fit_params: np.array) -> float:
        """
        A function that finds the accuracy at the inflection point of the fit function. Useful if you do not care
        about the specific threshold accuracy, but rather about e.g. the elevation at the inflection point.
        """
        max_fit_accuracy = self.fit_function(np.max(x_points), *fit_params)
        min_fit_accuracy = self.fit_function(np.min(x_points), *fit_params)
        threshold_accuracy = min_fit_accuracy + (max_fit_accuracy - min_fit_accuracy) / 2
        return threshold_accuracy

    @staticmethod
    def individual_score(source: float, target: Union[list, PropertyAssembly]) -> Score:
        """
        Computes the average distance of the source from each of the individual targets in units of the
        individual targets. This is generally a more stringent scoring method than pool_score, aimed
        to measure the average of the individual target effects.
        """
        raw_scores = []
        for target_value in target:
            raw_score = max((1 - ((np.abs(target_value - source)) / target_value)), 0)
            raw_scores.append(raw_score)

        raw_score, model_error = np.mean(raw_scores), np.std(raw_scores)
        raw_score = Score([raw_score, model_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        return raw_score

    @staticmethod
    def pool_score(source: float, target: Union[list, PropertyAssembly]) -> Score:
        """
        Computes the distance of the source from the average of the target in units of the target average.
        This is generally a less stringent scoring method than individual_score, aimed to measure the average
        target effect.
        """
        if not isinstance(target, list):
            target_mean = np.mean(target.values)
        else:
            target_mean = np.mean(target)
        raw_score = max((1 - ((np.abs(target_mean - source)) / target_mean)), 0)
        raw_score = Score([raw_score], coords={'aggregation': ['center', ]}, dims=['aggregation'])
        return raw_score

    @staticmethod
    def convert_proba_to_correct(source: BehavioralAssembly) -> np.array:
        """Converts the probability values returned by models doing probability tasks to behavioral choices."""
        decisions = np.argmax(source.values, axis=1)
        correct = []
        for presentation, decision in enumerate(decisions):
            if source['choice'].values[decision] == source['image_label'].values[presentation]:
                correct.append(1)
            else:
                correct.append(0)
        return np.array(correct)


class ThresholdElevation(Threshold):
    """
    Computes a threshold elevation from two conditions: a baseline condition and a test condition by dividing
    the threshold of the test condition by the baseline condition. In other words,

    `threshold_elevation = test_condition_threshold / baseline_condition_threshold`.
    """
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
                 source: Union[float, Dict[str, BehavioralAssembly]],
                 target: Union[list, Dict[str, PropertyAssembly]]
                 ) -> Score:
        """
        :param source: Either a dictionary containing the BehavioralAssemblies for the test condition and the
                        baseline condition, or a pre-computed float threshold elevation. If Dict, Dict
                        keys should be 'condition_assembly' and 'baseline_assembly' respectively.
        :param target: Either a dictionary containing the PropertyAssemblies for the test condition and the
                        baseline condition, or a list of pre-computed threshold elevations. If Dict, Dict
                        keys should be 'condition_assembly' and 'baseline_assembly' respectively.
        :return: A score containing the evaluated model's ceiling-adjusted distance to target threshold elevations.
        """
        # check whether source is a threshold elevation already - if not, compute it.
        if isinstance(source, float):
            raw_source_threshold_elevation = source
        elif isinstance(source, Dict):
            source_baseline_threshold = self.baseline_threshold_metric.compute_threshold(source[self.baseline_condition],
                                                                                         self._independent_variable)
            # if using the inflection accuracy, get the inflection point from the baseline condition, and use that
            # for the test condition.
            if self.threshold_accuracy == 'inflection':
                self.test_threshold_metric.threshold_accuracy = self.baseline_threshold_metric.threshold_accuracy
            source_test_threshold = self.test_threshold_metric.compute_threshold(source[self.test_condition],
                                                                                 self._independent_variable)
            raw_source_threshold_elevation = source_test_threshold / source_baseline_threshold
        else:
            raise TypeError(f'source is type {type(source)}, but type BehavioralAssembly or float is required.')

        # check whether the targets are threshold elevations already - if not, compute them
        if isinstance(target, list):
            target_threshold_elevations = target
        elif isinstance(target, Dict):
            target_threshold_elevations = self.compute_threshold_elevations(target)
        else:
            raise TypeError(f'target is type {type(target)}, but type PropertyAssembly or list is required.')

        # compare threshold elevation to target threshold elevations
        if self.scoring == 'pool':
            return self.pool_score(raw_source_threshold_elevation, target_threshold_elevations)
        elif self.scoring == 'individual':
            return self.individual_score(raw_source_threshold_elevation, target_threshold_elevations)
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
        """
        human_threshold_elevations = self.compute_threshold_elevations(assemblies)
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

    @staticmethod
    def compute_threshold_elevations(assemblies: Dict[str, PropertyAssembly]) -> list:
        """
        Computes the threshold elevations of a baseline condition and a test condition:

        `threshold_elevation = test_condition_threshold / baseline_condition_threshold`.
        """
        baseline_assembly = assemblies['baseline_assembly']
        condition_assembly = assemblies['condition_assembly']
        threshold_elevations = []
        for i, baseline_threshold in baseline_assembly.values:
            condition_threshold = condition_assembly[i]
            threshold_elevations.append(condition_threshold / baseline_threshold)
        return threshold_elevations
