import numpy as np

from mkgu.assemblies import DataAssembly
from mkgu.metrics import NonparametricCVMetric


class RSA(object):
    """
    Representational Similarity Matrix

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __call__(self, assembly):
        correlations = np.corrcoef(assembly) if assembly.dims[-1] == 'neuroid' else np.corrcoef(assembly.T).T
        coords = {coord: coord_value for coord, coord_value in assembly.coords.items() if coord != 'neuroid'}
        dims = [dim if dim != 'neuroid' else assembly.dims[(i - 1) % len(assembly.dims)]
                for i, dim in enumerate(assembly.dims)]
        return DataAssembly(correlations, coords=coords, dims=dims)


class RDM(RSA):
    """
    Representational Dissimilarity Matrix

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __call__(self, assembly):
        return 1 - super(RDM, self).__call__(assembly)


class RDMSimilarity(object):
    def __call__(self, rdm_assembly1, rdm_assembly2):
        triu1 = self._triangulars(rdm_assembly1.values)
        triu2 = self._triangulars(rdm_assembly2.values)
        corr = np.corrcoef(triu1, triu2)
        np.testing.assert_array_equal(corr.shape, [2, 2])
        return corr[0, 1]

    def _triangulars(self, values):
        assert len(values.shape) == 2 and values.shape[0] == values.shape[1]
        np.testing.assert_almost_equal(np.diag(values), 0)
        triangular_indices = np.triu_indices(values.shape[0], k=1)
        return values[triangular_indices]


class RDMMetric(NonparametricCVMetric):
    """
    Computes a coefficient for the similarity between two `RDM`s, using the upper triangular regions

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rdm = RDM()
        self._similarity = RDMSimilarity()

    def compute(self, rdm1, rdm2):
        """
        :param mkgu.assemblies.NeuroidAssembly rdm1:
        :param mkgu.assemblies.NeuroidAssembly rdm2:
        :param str similarity_dims: indicate the dimension along which the RDM/RSA was computed,
            either with a string for a repeated dimension or with a list for two different dimension names
        :return: mkgu.assemblies.DataAssembly
        """
        rdm1 = self._rdm(rdm1)
        rdm2 = self._rdm(rdm2)
        return self._similarity(rdm1, rdm2)
