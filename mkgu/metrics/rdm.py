import numpy as np

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

    def __init__(self, **kwargs):
        super(RSA, self).__init__(**kwargs)

    def apply(self, assembly):
        return np.corrcoef(assembly)


class RDM(RSA):
    """
    Representational Dissimilarity Matrix

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def apply(self, assembly):
        return 1 - super(RDM, self).apply(assembly)


class RDMCorrelationCoefficient(Similarity):
    """
    Computes a coefficient for the similarity between two `RDM`s, using the upper triangular regions

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def apply(self, assembly1, assembly2):
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
