from typing import Dict, Union, Tuple, Optional, Callable
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from brainscore.metrics import Metric, Score
from brainio.assemblies import PropertyAssembly, BehavioralAssembly


def psychometric_cum_gauss(x: np.array, alpha: float, beta: float, lambda_: float, gamma: float = 0.5) -> float:
    """
    The classic psychometric function as implemented in Wichmann & Hill (2001). The psychometric function: I.
    Fitting, sampling, and goodness of fit, eq. 1.

    :param x: the independent variables of the data
    :param alpha: the slope parameter
    :param beta: the mean of the cdf parameter
    :param lambda_: the lapse rate
    :param gamma: the lower bound of the fit

    :return: the psychometric function values for the given parameters evaluated at `x`.
    """
    return gamma + (1 - gamma - lambda_) * norm.cdf(alpha * (x - beta))


def inverse_psychometric_cum_gauss(y: np.array, alpha: float, beta: float, lambda_: float, gamma: float = 0.5) -> float:
    """The inverse of psychometric_cum_gauss."""
    return beta + (norm.ppf((y - gamma) / (1 - gamma - lambda_)) / alpha)


def cum_gauss_neg_log_likelihood(params: Tuple[float, ...], x: np.array, y: np.array) -> float:
    """The negative log likelihood function for psychometric_cum_gauss."""
    alpha, beta, lambda_ = params
    p = psychometric_cum_gauss(x, alpha, beta, lambda_)
    log_likelihood = y * np.log(p) + (1 - y) * np.log(1 - p)
    return -np.sum(log_likelihood)


def get_predicted(params: Tuple[float, ...], x: np.array, fit_fn: Callable) -> np.array:
    """Returns the predicted values based on the model parameters."""
    return fit_fn(x, *params)


def grid_search(x: np.array,
                y: np.array,
                alpha_values: np.array = np.logspace(-3, 1, 50),
                beta_values: np.array = None,
                fit_fn: Callable = psychometric_cum_gauss,
                fit_log_likelihood_fn: Callable = cum_gauss_neg_log_likelihood,
                fit_bounds: Tuple = ((None, None), (None, None), (0.03, 0.5))
                ) -> Tuple[Tuple[float, ...], float]:
    """
    A classic simplified procedure for running sparse grid search over the slope and mean parameters of the
    psychometric function.
    This function is implemented here instead of using sklearn.GridSearchCV since we would have to make a custom
    sklearn estimator class to use GridSearchCV with psychometric functions, likely increasing code bloat
    substantially.

    :param x: the independent variables of the data
    :param y: the measured accuracy rates for the given x-values
    :param alpha_values: the alpha values for the chosen fit function to grid search over
    :param beta_values: the beta values for the chosen fit function to grid search over
    :param fit_fn: the psychometric function that is fit
    :param fit_log_likelihood_fn: the log likelihood function that computes the log likelihood of its corresponding
                                  fit function
    :param fit_bounds: the bounds assigned to the fit function called by fit_log_likelihood_fn.
                       The default fit_bounds are assigned as:
                       alpha: (None, None), to allow any slope
                       beta: (None, None), any inflection point is allowed, as that is controlled for in the
                             Threshold class
                       lambda_: (0.03, 0.5)), to require at least a small lapse rate, as is regularly done in
                                human fitting

    :return: the parameters of the best fit in the grid search
    """
    assert len(x) == len(y)
    # Default the beta_values grid search to the measured x-points.
    if beta_values is None:
        beta_values = x

    # initialize best values for a fit
    best_alpha, best_beta, best_lambda = None, None, None
    min_neg_log_likelihood = np.inf

    for alpha_guess in alpha_values:
        for beta_guess in beta_values:
            initial_guess = np.array([alpha_guess, beta_guess, 1 - np.max(y)])  # lapse rate guess set to the maximum y

            # wrap inside a RuntimeError block to catch the RuntimeError thrown by scipy.minimize if a fit
            # entirely fails. The case where all fits fail here is handled by the Threshold metric.
            try:
                result = minimize(fit_log_likelihood_fn, initial_guess, args=(x, y),
                                  method='L-BFGS-B', bounds=fit_bounds)
                alpha_hat, beta_hat, lambda_hat = result.x
                neg_log_likelihood_hat = fit_log_likelihood_fn([alpha_hat, beta_hat, lambda_hat], x, y)

                if neg_log_likelihood_hat < min_neg_log_likelihood:
                    min_neg_log_likelihood = neg_log_likelihood_hat
                    best_alpha, best_beta, best_lambda = alpha_hat, beta_hat, lambda_hat
            except RuntimeError:
                pass

    y_pred = fit_fn(x, best_alpha, best_beta, best_lambda)
    r2 = r2_score(y, y_pred)  # R^2 of the fit
    return (best_alpha, best_beta, best_lambda), r2


class Threshold(Metric):
    """
    Computes a psychometric threshold function from model responses and compares against human-computed psychometric
    thresholds.

    The model comparison to human data is currently individual-subject based, i.e., models and ceilings are compared
    against the mean of the distance of the model threshold to human thresholds.
    """
    def __init__(self,
                 independent_variable: str,
                 fit_function=psychometric_cum_gauss,
                 fit_inverse_function=inverse_psychometric_cum_gauss,
                 threshold_accuracy: Union[str, float] = 'inflection',
                 scoring: str = 'pool',
                 required_accuracy: Optional[float] = 0.6,
                 plot_fit: bool = False
                 ):
        """
        :param independent_variable: The independent variable in the benchmark that the threshold is computed
                                      over.
        :param fit_function: The function used to fit the threshold.
        :param fit_inverse_function: The inverse of fit_function used to find the threshold from the fit.
        :param threshold_accuracy: The accuracy at which the threshold should be evaluated at. This can be
                                    either a string Literal['inflection'] or a float. When Literal['inflection']
                                    is used, the function finds the inflection point of the curve and evaluates
                                    the threshold at that level. When a float is used, the function evaluates
                                    the threshold at that level.
        :param scoring: The scoring function used to evaluate performance. Either Literal['individual'] or
                         Literal['pool']. See the individual_score and pool_score methods for more information.
        """
        self.fit_function = fit_function
        self.fit_inverse_function = fit_inverse_function
        self._independent_variable = independent_variable
        self.threshold_accuracy = threshold_accuracy
        self.scoring = scoring
        self.required_accuracy = required_accuracy
        self.plot_fit = plot_fit

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
                return Score([0., 0.], coords={'aggregation': ['center', ]}, dims=['aggregation'])
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
        Computed by one-vs all for each of the NUM_TRIALS human indexes. One index is removed, and scored against
        a pool of the other values.

        Currently copied with modification from 'https://github.com/brain-score/brain-score/blob/
        jacob2020_occlusion_depth_ordering/brainscore/metrics/data_cloud_comparision.py#L54'.

        :param assembly: the human PropertyAssembly containing human responses, or a dict containing the
                          PropertyAssemblies of the ThresholdElevation metric.
        :return: Score object with coords center (ceiling) and error (STD)
        """
        human_thresholds: list = assembly.values.tolist()
        scores = []
        for i in range(len(human_thresholds)):
            random_state = np.random.RandomState(i)
            random_human_score = random_state.choice(human_thresholds, replace=False)
            metric = Threshold(self._independent_variable, self.fit_function, self.fit_inverse_function,
                               self.threshold_accuracy, scoring=self.scoring)
            human_thresholds.remove(random_human_score)
            score = metric(random_human_score, human_thresholds)
            score = float(score[(score['aggregation'] == 'center')].values)
            human_thresholds.append(random_human_score)
            scores.append(score)

        ceiling, ceiling_error = np.mean(scores), np.std(scores)
        ceiling = Score([ceiling, ceiling_error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        return ceiling

    def compute_threshold(self, source: BehavioralAssembly, independent_variable: str) -> Union[float, str]:
        """Converts the source BehavioralAssembly to a threshold float value."""
        assert len(source.values) == len(source[independent_variable].values)

        x_points = source[independent_variable].values
        accuracies = self.convert_proba_to_correct(source)
        if np.mean(accuracies) < self.required_accuracy:
            print('Psychometric threshold fit failure due to low accuracy.')
            fit_params = 'fit_fail'
        else:
            fit_params, measurement_max = self.fit_threshold_function(x_points, accuracies)
        if (type(fit_params) == str) and (fit_params == 'fit_fail'):
            return fit_params

        if self.threshold_accuracy == 'inflection':
            self.threshold_accuracy = self.inflection_accuracy(x_points, fit_params)

        threshold = self.find_threshold(self.threshold_accuracy, fit_params)

        # check whether the fit is outside the measured model responses to discard spurious thresholds
        if (threshold > measurement_max) or np.isnan(threshold):
            print('Fit fail because threshold is outside of the measured range of responses.')
            return 'fit_fail'
        return threshold

    def fit_threshold_function(self, x_points: np.array, y_points: np.array) -> Union[np.array, str]:
        """
        A function that takes the x and y-points of the measured variable and handles the fitting of the
        psychometric threshold function.

        Returns either the fit parameters for self.fit_function or a string tag that indicates the failure
        of the psychometric curve fit.
        """
        x_points, y_points = self.aggregate_psychometric_fit_data(x_points, y_points)
        aggregated_x_points, aggregated_y_points, at_least_third_remaining = self.remove_data_after_asymptote(x_points,
                                                                                                              y_points)
        measurement_max = np.max(aggregated_x_points)
        if not at_least_third_remaining:
            # This failure indicates that there is too little data to accurately fit the psychometric function.
            print('Psychometric curve fit fail because performance is decreasing with the independent variable.')
            return 'fit_fail', measurement_max

        params, r2 = grid_search(aggregated_x_points, aggregated_y_points)

        # if all the fits in the grid search failed, there will be a None value in params. In this case, we reject
        #  the fit. This typically only ever happens when a model outputs one value for all test images.
        if None in params:
            params = 'fit_fail'

        # remove fits to random data. This choice is preferred over a chi^2 test since chi^2 discards a lot of fits
        #  that would be acceptable in a human case.
        if r2 < 0.4:
            print('Fit fail due to low fit R^2.')
            params = 'fit_fail'

        if self.plot_fit:
            self.plot_fit_(x_points,
                           aggregated_x_points,
                           y_points,
                           aggregated_y_points,
                           params,
                           fit_function=self.fit_function)
        return params, measurement_max

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

    def plot_fit_(self, x_points, x_points_removed, y_points, y_points_removed, fit_params, fit_function):
        # Create a dense set of x values for plotting the fitted curve
        x_dense = np.linspace(min(x_points), max(x_points), 1000)
        # Calculate the corresponding y values using the fit function and parameters
        y_dense = fit_function(x_dense, *fit_params)

        # Plot the original data points
        plt.scatter(x_points, y_points, label='Before asymptote removal',
                    marker='o', color='blue', alpha=0.5)
        plt.scatter(x_points_removed, y_points_removed, label='After asymptote removal',
                    marker='o', color='red', alpha=0.5)

        # Plot the fitted curve
        plt.plot(x_dense, y_dense, label='Fitted curve', color='red', linewidth=2)

        # Add labels and a legend
        plt.xlabel(self._independent_variable)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    @staticmethod
    def aggregate_psychometric_fit_data(x_points, y_points):
        unique_x = np.unique(x_points)
        correct_rate = np.zeros(len(unique_x))

        for i, x in enumerate(unique_x):
            trials = np.sum(x_points == x)
            correct_trials = np.sum((x_points == x) & (y_points == 1))
            correct_rate[i] = correct_trials / trials

        return unique_x, correct_rate

    @staticmethod
    def individual_score(source: float, target: Union[list, PropertyAssembly]) -> Score:
        """
        Computes the average distance of the source from each of the individual targets in units of the
        individual targets. This is generally a more stringent scoring method than pool_score, aimed
        to measure the average of the individual target effects.
        """
        raw_scores = []
        for target_value in target:
            # This score = 0 when the source exceeds target_value by 100%
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
        # This score = 0 when the source exceeds target_mean by 100%
        raw_score = max((1 - ((np.abs(target_mean - source)) / target_mean)), 0)
        raw_score = Score([raw_score], coords={'aggregation': ['center']}, dims=['aggregation'])
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

    @staticmethod
    def remove_data_after_asymptote(x_values, y_values):
        """
        A function that removes all data after the point at which all values of the measured variable are 1 standard
        deviation less than the maximum.

        This is done to simulate the procedure in which an experimenter fine-tunes the stimuli in a pilot experiment
        to the given system (e.g., humans) such that they only measure data in a region within which the psychometric
        fit is monotone (as per the function fit assumption). When this assumption is violated, the function fit
        is not a valid measure of the underlying performance function.

        There are circumstances in which this behavior is expected (e.g., crowding). When e.g. a vernier element's
        offset is increased enough, the task may paradoxically become more difficult, as the offset grows large
        enough such that the relevant elements do not fall within a spatially relevant window, or group with the
        flankers more than with each other due to constant target-flanker distance.
        """

        std_dev = np.std(y_values)
        max_y_idx = np.argmax(y_values)

        # initialize the index for the first data point after the maximum y_value
        #  that deviates from the maximum by at least 1 standard deviation
        index_to_remove = None

        # iterate through the y_values after the maximum y_value
        for idx, y in enumerate(y_values[max_y_idx + 1:], start=max_y_idx + 1):
            # check if all the remaining y_values deviate by at least 1 standard deviation
            if all([abs(val - y_values[max_y_idx]) >= std_dev for val in y_values[idx:]]):
                index_to_remove = idx
                break
        pre_remove_length = len(y_values)
        # if we found an index to remove, remove the data after that index
        if index_to_remove is not None:
            x_values = x_values[:index_to_remove]
            y_values = y_values[:index_to_remove]

        # check if at least a third of the elements remain. This is done so that we have an adequate amount of data
        #  to fit a psychometric threshold on.
        remaining_fraction = len(y_values) / pre_remove_length
        is_at_least_third_remaining = remaining_fraction >= 1 / 3

        return x_values, y_values, is_at_least_third_remaining


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
                 threshold_accuracy: Union[str, float] = 'inflection',
                 scoring: str = 'pool',
                 required_baseline_accuracy: Optional[float] = 0.6,
                 required_test_accuracy: Optional[float] = 0.6,
                 plot_fit: bool = False
                 ):
        """
        :param independent_variable: The independent variable in the benchmark that the threshold is computed
                                      over.
        :param baseline_condition: The baseline condition against which threshold elevation is measured.
        :param test_condition: The test condition that is used to measure threshold elevation..
        :param threshold_accuracy: The accuracy at which the threshold should be evaluated at. This can be
                                    either a string Literal['inflection'] or a float. When Literal['inflection']
                                    is used, the function finds the inflection point of the curve and evaluates
                                    the threshold at that level. When a float is used, the function evaluates
                                    the threshold at that level.
        :param scoring: The scoring function used to evaluate performance. Either Literal['individual'] or
                         Literal['pool']. See the individual_score and pool_score methods for more information.
        """
        super(ThresholdElevation, self).__init__(independent_variable)
        self.baseline_threshold_metric = Threshold(self._independent_variable,
                                                   threshold_accuracy=threshold_accuracy,
                                                   required_accuracy=required_baseline_accuracy,
                                                   plot_fit=plot_fit)
        self.test_threshold_metric = Threshold(self._independent_variable,
                                               threshold_accuracy=threshold_accuracy,
                                               required_accuracy=required_test_accuracy,
                                               plot_fit=plot_fit)
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
            if source_baseline_threshold == 'fit_fail' or source_test_threshold == 'fit_fail':
                return Score([0., 0.], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
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

    def ceiling(self, assemblies: Dict[str, PropertyAssembly]) -> Score:
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
                                        self.threshold_accuracy, scoring=self.scoring)
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
        for i, baseline_threshold in enumerate(baseline_assembly.values):
            condition_threshold = condition_assembly.values[i]
            threshold_elevations.append(condition_threshold / baseline_threshold)
        return threshold_elevations
