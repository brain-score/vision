import numpy as np

import mkgu
from mkgu.metrics import Characterization, Metric, Similarity


class RDMMetric(Metric):
    """
    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __init__(self):
        super(RDMMetric, self).__init__(characterization=RDM(),
                                        similarity=RDMCorrelationCoefficient())


class RSA(Characterization):
    """
    Representational Similarity Matrix

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __call__(self, assembly):
        correlations = np.corrcoef(assembly) if assembly.dims[-1] == 'neuroid' else np.corrcoef(assembly.T).T
        coords = {coord: coord_value for coord, coord_value in assembly.coords.items() if coord != 'neuroid'}
        dims = [dim if dim != 'neuroid' else assembly.dims[0] for dim in assembly.dims]
        return mkgu.assemblies.NeuroidAssembly(correlations, coords=coords, dims=dims)


class RDM(RSA):
    """
    Representational Dissimilarity Matrix

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __call__(self, assembly):
        return 1 - super(RDM, self).__call__(assembly)


class RDMCorrelationCoefficient(Similarity):
    """
    Computes a coefficient for the similarity between two `RDM`s, using the upper triangular regions

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __call__(self, assembly1, assembly2):
        corr = np.corrcoef(self._preprocess_assembly(assembly1), self._preprocess_assembly(assembly2))
        assert len(corr.shape) == 2 and corr.shape[0] == 2 and corr.shape[1] == 2
        assert all(np.diag(corr) == 1)
        assert corr[0, 1] == corr[1, 0]
        return corr[0, 1]

    def _preprocess_assembly(self, assembly):
        assert len(assembly.shape) == 2
        assert assembly.shape[0] == assembly.shape[1]
        assert np.allclose(np.diag(assembly), np.zeros((assembly.shape[0],)), atol=1e-10)
        triangular_indices = np.triu_indices(assembly.shape[0], k=-1)
        return assembly[triangular_indices]
