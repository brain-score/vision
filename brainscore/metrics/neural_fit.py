import logging
from abc import ABCMeta

import scipy.stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA as PCAImpl
from sklearn.linear_model import LinearRegression

import brainscore
from brainscore.assemblies import NeuroidAssembly, DataAssembly
from brainscore.assemblies import array_is_element, walk_coords
from brainscore.metrics import ParametricMetric
from brainscore.utils import fullname


class ParametricNeuroidMetric(ParametricMetric):
    class Defaults:
        expected_dims = ['presentation', 'neuroid']
        neuroid_dim = 'neuroid'
        neuroid_coord = 'neuroid_id'
        correlation = scipy.stats.pearsonr

    def __init__(self, *args, neuroid_dim=Defaults.neuroid_dim, neuroid_coord=Defaults.neuroid_coord,
                 correlation=Defaults.correlation, **kwargs):
        kwargs = {**dict(expected_source_dims=self.Defaults.expected_dims,
                         expected_target_dims=self.Defaults.expected_dims),
                  **kwargs}
        super().__init__(*args, **kwargs)
        self._neuroid_dim = neuroid_dim
        self._neuroid_coord = neuroid_coord
        self._correlation = correlation
        self._target_neuroid_values = None

    def fit(self, source, target):
        super(ParametricNeuroidMetric, self).fit(source, target)
        self._target_neuroid_values = {}
        for name, dims, values in walk_coords(target):
            if self._neuroid_dim in dims:
                assert array_is_element(dims, self._neuroid_dim)
                self._target_neuroid_values[name] = values

    def package_prediction(self, predicted_values, source):
        coords = {coord: (dims, values) for coord, dims, values in walk_coords(source)
                  if not array_is_element(dims, self._neuroid_dim)}
        # re-package neuroid coords
        for target_coord, target_value in self._target_neuroid_values.items():
            coords[target_coord] = self._neuroid_dim, target_value  # this might overwrite values which is okay
        prediction = NeuroidAssembly(predicted_values, coords=coords, dims=source.dims)
        return prediction

    def compare_prediction(self, prediction, target):
        assert all(prediction[self._stimulus_coord].values == target[self._stimulus_coord].values)
        assert all(prediction[self._neuroid_coord].values == target[self._neuroid_coord].values)
        correlations = []
        for i in target[self._neuroid_coord].values:
            target_activations = target.sel(**{self._neuroid_coord: i}).squeeze()
            prediction_activations = prediction.sel(**{self._neuroid_coord: i}).squeeze()
            r, p = self._correlation(target_activations, prediction_activations)
            correlations.append(r)
        return DataAssembly(correlations, coords={self._neuroid_coord: target[self._neuroid_coord].values},
                            dims=[self._neuroid_coord])


class RegressionFit(ParametricNeuroidMetric, metaclass=ABCMeta):
    def __init__(self, *args, regression, **kwargs):
        super().__init__(*args, **kwargs)
        self._regression = regression

    def fit_values(self, source, target):
        self._regression.fit(source, target)

    def predict_values(self, test_source):
        return self._regression.predict(test_source)

    def aggregate(self, scores, neuroid_dim='neuroid_id'):
        return scores.median(dim=neuroid_dim)


class NeuralFit(RegressionFit):
    """
    Yamins & Hong et al., 2014 https://doi.org/10.1073/pnas.1403112111
    """

    regressions = {
        'pls-25': PLSRegression(n_components=25, scale=False),
    }

    def __init__(self):
        super().__init__(regression=LinearRegression())


class Pls25Fit(RegressionFit):
    """
    Schrimpf & Kubilius et al., 2018
    """

    def __init__(self):
        super().__init__(regression=PLSRegression(n_components=25, scale=False))


class PCA(object):
    def __init__(self, max_components):
        self._pca = PCAImpl(n_components=max_components)
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, assembly):
        assert len(assembly.neuroid.shape) == 1
        if assembly.neuroid.shape[0] <= self._pca.n_components:
            return assembly
        self._logger.debug('PCA from {} to {}'.format(assembly.neuroid.shape[0], self._pca.n_components))
        transformed_values = self._pca.fit_transform(assembly)

        coords = {dim if dim != 'neuroid' else 'neuroid_components':
                      assembly[dim] if dim != 'neuroid' else assembly.neuroid[:self._pca.n_components]
                  for dim in assembly.coords}
        return brainscore.assemblies.NeuroidAssembly(transformed_values, coords=coords, dims=assembly.dims)
