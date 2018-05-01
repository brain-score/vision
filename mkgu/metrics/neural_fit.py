import logging

import numpy as np
import scipy
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

import mkgu
from mkgu.assemblies import NeuroidAssembly
from mkgu.metrics import Metric, Characterization, ParametricCVSimilarity, Score
from mkgu.metrics.utils import get_coords


class NeuralFitMetric(Metric):
    def __init__(self):
        super(NeuralFitMetric, self).__init__(similarity=NeuralFitSimilarity())


class NeuralFitSimilarity(ParametricCVSimilarity):
    """
    Yamins & Hong et al., 2014 https://doi.org/10.1073/pnas.1403112111
    """

    def __init__(self, num_splits=10, test_size=.25, regression_components=25):
        super(NeuralFitSimilarity, self).__init__()
        self._split_strategy = StratifiedShuffleSplit(n_splits=num_splits, test_size=test_size)
        self._regression = PLSRegression(n_components=regression_components, scale=False)
        self._target_neuroid_ids = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def score(self, similarity_assembly):
        return MedianScore(similarity_assembly)

    def fit(self, train_source, train_target):
        self._logger.debug("Fitting")
        np.testing.assert_array_equal(train_source.dims, ['presentation', 'neuroid'])
        np.testing.assert_array_equal(train_target.dims, ['presentation', 'neuroid'])
        self._target_neuroid_ids = train_target['neuroid_id']
        self._regression.fit(train_source, train_target)

    def predict(self, test_source):
        self._logger.debug("Predicting")
        np.testing.assert_array_equal(test_source.dims, ['presentation', 'neuroid'])
        predicted_values = self._regression.predict(test_source)

        def modify_coord(name, dims, values):
            if name == 'neuroid_id':
                values = self._target_neuroid_ids
            return name, (dims, values)

        coords = get_coords(test_source, modify_coord)
        return NeuroidAssembly(predicted_values, coords=coords, dims=test_source.dims)

    def compare_prediction(self, prediction, target, axis='neuroid_id', correlation=scipy.stats.pearsonr):
        self._logger.debug("Comparing")
        assert all(target[axis] == prediction[axis])
        rs = []
        for i in target[axis].values:
            d1 = target.sel(**{axis: i})
            d2 = prediction.sel(**{axis: i})
            r, p = correlation(d1, d2)
            rs.append(r)
        return np.mean(rs)


class MedianScore(Score):
    def get_center(self, values, dim):
        return values.median(dim=dim)


class PCANeuroidCharacterization(Characterization):
    def __init__(self, max_components):
        self._pca = PCA(n_components=max_components)
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
