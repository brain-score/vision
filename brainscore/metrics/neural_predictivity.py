import scipy.stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

from brainscore.metrics.transformations import Transformations, CrossValidation
from brainscore.metrics.xarray_utils import XarrayRegression, XarrayCorrelation, Defaults as XarrayDefaults


class PlsPredictivity:
    def __init__(self, n_components=25,
                 expected_dims=XarrayDefaults.expected_dims, neuroid_dim=XarrayDefaults.neuroid_dim,
                 stimulus_coord=XarrayDefaults.stimulus_coord, neuroid_coord=XarrayDefaults.neuroid_coord):
        self._predictor = pls_predictor(n_components=n_components, expected_dims=expected_dims, neuroid_dim=neuroid_dim,
                                        stimulus_coord=stimulus_coord, neuroid_coord=neuroid_coord)
        cross_validation = CrossValidation()
        self._transformations = Transformations([cross_validation])

    def __call__(self, source, target):
        return self._transformations(source, target, metric=self._predictor)


def pls_predictor(n_components=25,
                  expected_dims=XarrayDefaults.expected_dims, neuroid_dim=XarrayDefaults.neuroid_dim,
                  stimulus_coord=XarrayDefaults.stimulus_coord, neuroid_coord=XarrayDefaults.neuroid_coord):
    regression = PLSRegression(n_components=n_components, scale=False)
    correlation = scipy.stats.pearsonr
    return RegressionPredictor(regression=regression, correlation=correlation,
                               expected_dims=expected_dims, stimulus_coord=stimulus_coord,
                               neuroid_dim=neuroid_dim, neuroid_coord=neuroid_coord)


class LinearPredictivity:
    def __init__(self, expected_dims=XarrayDefaults.expected_dims, neuroid_dim=XarrayDefaults.neuroid_dim,
                 stimulus_coord=XarrayDefaults.stimulus_coord, neuroid_coord=XarrayDefaults.neuroid_coord):
        self._predictor = linear_predictor(expected_dims=expected_dims, stimulus_coord=stimulus_coord,
                                           neuroid_dim=neuroid_dim, neuroid_coord=neuroid_coord)
        cross_validation = CrossValidation()
        self._transformations = Transformations([cross_validation])

    def __call__(self, source, target):
        return self._transformations(source, target, metric=self._predictor)


def linear_predictor(expected_dims=XarrayDefaults.expected_dims, neuroid_dim=XarrayDefaults.neuroid_dim,
                     stimulus_coord=XarrayDefaults.stimulus_coord, neuroid_coord=XarrayDefaults.neuroid_coord):
    regression = LinearRegression()
    correlation = scipy.stats.pearsonr
    return RegressionPredictor(regression=regression, correlation=correlation,
                               expected_dims=expected_dims, stimulus_coord=stimulus_coord,
                               neuroid_dim=neuroid_dim, neuroid_coord=neuroid_coord)


class RegressionPredictor:
    """
    Helper class to fit a regression on data, predict held out data and compare predictions.
    """

    def __init__(self, regression, correlation,
                 expected_dims=XarrayDefaults.expected_dims, neuroid_dim=XarrayDefaults.neuroid_dim,
                 stimulus_coord=XarrayDefaults.stimulus_coord, neuroid_coord=XarrayDefaults.neuroid_coord):
        self._neuroid_dim = neuroid_dim
        self._regression = XarrayRegression(regression, expected_dims=expected_dims, neuroid_dim=neuroid_dim,
                                            neuroid_coord=neuroid_coord)
        self._correlation = XarrayCorrelation(correlation, stimulus_coord=stimulus_coord, neuroid_coord=neuroid_coord)
        self._target_neuroid_values = None

    def __call__(self, source_train, target_train, source_test, target_test):
        self._regression.fit(source_train, target_train)
        prediction = self._regression.predict(source_test)
        score = self._correlation(prediction, target_test)
        return score

    def aggregate(self, scores):  # TODO: need to decide where this goes
        return scores.median(dim=self._neuroid_dim)
