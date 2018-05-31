import logging

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA as PCAImpl
from sklearn.linear_model import LinearRegression

import mkgu
from mkgu.assemblies import NeuroidAssembly
from mkgu.metrics import ParametricMetric


class NeuralFit(ParametricMetric):
    """
    Yamins & Hong et al., 2014 https://doi.org/10.1073/pnas.1403112111
    """

    regressions = {
        'pls-25': PLSRegression(n_components=25, scale=False),
        'linear': LinearRegression(),
    }

    def __init__(self, *args, regression='pls-25', **kwargs):
        super().__init__(*args, **kwargs)
        self._regression = self.regressions[regression] if isinstance(regression, str) else regression

    def fit_values(self, train_source, train_target):
        self._regression.fit(train_source, train_target)

    def predict_values(self, test_source):
        return self._regression.predict(test_source)


class PCA(object):
    def __init__(self, max_components):
        self._pca = PCAImpl(n_components=max_components)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, assembly):
        assert len(assembly.neuroid.shape) == 1
        if assembly.neuroid.shape[0] <= self._pca.n_components:
            return assembly
        self._logger.debug('PCA from {} to {}'.format(assembly.neuroid.shape[0], self._pca.n_components))
        transformed_values = self._pca.fit_transform(assembly)

        coords = {dim if dim != 'neuroid' else 'neuroid_components':
                      assembly[dim] if dim != 'neuroid' else assembly.neuroid[:self._pca.n_components]
                  for dim in assembly.coords}
        return mkgu.assemblies.NeuroidAssembly(transformed_values, coords=coords, dims=assembly.dims)
