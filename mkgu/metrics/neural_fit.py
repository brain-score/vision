import logging

import numpy as np
import scipy
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

import mkgu
from mkgu.metrics import Metric, Similarity, Characterization


class NeuralFitMetric(Metric):
    def __init__(self, pca_components):
        characterization = PCANeuroidCharacterization(pca_components) if pca_components is not None else None
        super(NeuralFitMetric, self).__init__(characterization=characterization,
                                              similarity=NeuralFitSimilarity())


class NeuralFitSimilarity(Similarity):
    """
    Yamins & Hong et al., 2014 https://doi.org/10.1073/pnas.1403112111
    """

    def __init__(self, num_splits=10, test_size=.25, regression_components=25):
        super(NeuralFitSimilarity, self).__init__()
        self._split_strategy = StratifiedShuffleSplit(n_splits=num_splits, test_size=test_size)
        self._regression = PLSRegression(n_components=regression_components, scale=False)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, source_assembly, target_assembly):
        assert source_assembly.shape[0] == target_assembly.shape[0]  # same stimuli
        source_assembly = source_assembly.sortby(source_assembly.id)
        target_assembly = target_assembly.sortby(target_assembly.id)
        np.testing.assert_array_equal(target_assembly.obj.values, source_assembly.obj.values)
        object_labels = source_assembly.obj

        correlations = []
        for split_iterator, (train_indices, test_indices) in enumerate(
                self._split_strategy.split(np.zeros(len(object_labels)), object_labels.values)):
            # fit
            self._logger.debug('Fitting split {}/{}'.format(split_iterator + 1, self._split_strategy.n_splits))
            self._regression.fit(source_assembly[train_indices], target_assembly[train_indices])
            predicted_responses = self._regression.predict(source_assembly[test_indices])
            # correlate
            self._logger.debug('Correlating split {}/{}'.format(split_iterator + 1, self._split_strategy.n_splits))
            rs = pearsonr_matrix(target_assembly[test_indices].values, predicted_responses)
            correlations.append(rs)

        return np.mean(correlations)


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

        coords = {coord if coord != 'neuroid' else 'neuroid_components':
                      assembly[coord] if coord != 'neuroid' else assembly.neuroid[:self._pca.n_components]
                  for coord in assembly.coords}
        return mkgu.assemblies.NeuroidAssembly(transformed_values, coords=coords, dims=assembly.dims)


def pearsonr_matrix(data1, data2, axis=1):
    rs = []
    for i in range(data1.shape[axis]):
        d1 = np.take(data1, i, axis=axis)
        d2 = np.take(data2, i, axis=axis)
        r, p = scipy.stats.pearsonr(d1, d2)
        rs.append(r)
    return np.array(rs)
