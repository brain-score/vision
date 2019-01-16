import numpy as np

from brainio_base.assemblies import DataAssembly
from brainscore.metrics.transformations import CrossValidation
from brainscore.metrics.xarray_utils import Defaults as XarrayDefaults


class RDMCrossValidated:
    """
    Computes a coefficient for the similarity between two `RDM`s, using the upper triangular regions

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __init__(self, neuroid_dim=XarrayDefaults.neuroid_dim, comparison_coord=XarrayDefaults.stimulus_coord):
        self._metric = RDMMetric(neuroid_dim=neuroid_dim, comparison_coord=comparison_coord)
        self._cross_validation = CrossValidation(test_size=.9)  # leave 10% out

    class LeaveOneOutWrapper:
        def __init__(self, metric):
            self._metric = metric

        def __call__(self, train_source, train_target, test_source, test_target):
            # compare assemblies for a single split. we ignore the 10% train ("leave-one-out") and only use test.
            return self._metric(test_source, test_target)

    def __call__(self, assembly1, assembly2):
        """
        :param brainscore.assemblies.NeuroidAssembly assembly1:
        :param brainscore.assemblies.NeuroidAssembly assembly2:
        :return: brainscore.assemblies.DataAssembly
        """

        leave_one_out = self.LeaveOneOutWrapper(self._metric)
        return self._cross_validation(assembly1, assembly2, apply=leave_one_out)


class RDMMetric:
    """
    Computes a coefficient for the similarity between two `RDM`s, using the upper triangular regions

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __init__(self, neuroid_dim=XarrayDefaults.neuroid_dim, comparison_coord=XarrayDefaults.stimulus_coord):
        self._neuroid_dim = neuroid_dim
        self._rdm = RDM(neuroid_dim=neuroid_dim)
        self._similarity = RDMSimilarity(comparison_coord=comparison_coord)

    def __call__(self, assembly1, assembly2):
        """
        :param brainscore.assemblies.NeuroidAssembly assembly1:
        :param brainscore.assemblies.NeuroidAssembly assembly2:
        :return: brainscore.assemblies.DataAssembly
        """

        rdm1 = self._rdm(assembly1)
        rdm2 = self._rdm(assembly2)
        similarity = self._similarity(rdm1, rdm2)
        return DataAssembly(similarity)


class RSA:
    """
    Representational Similarity Analysis.
    Converts an assembly of `presentation x neuroid` into a `neuroid x neuroid` RSA matrix.

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __init__(self, neuroid_dim=XarrayDefaults.neuroid_dim):
        self._neuroid_dim = neuroid_dim

    def __call__(self, assembly):
        assert len(assembly.dims) == 2
        correlations = np.corrcoef(assembly) if assembly.dims[-1] == self._neuroid_dim else np.corrcoef(assembly.T).T
        coords = {coord: coord_value for coord, coord_value in assembly.coords.items() if coord != self._neuroid_dim}
        dims = [dim if dim != self._neuroid_dim else assembly.dims[(i - 1) % len(assembly.dims)]
                for i, dim in enumerate(assembly.dims)]
        return DataAssembly(correlations, coords=coords, dims=dims)


class RDM:
    """
    Representational Dissimilarity Matrix.
    Converts an assembly of `presentation x neuroid` into a `neuroid x neuroid` RDM.

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __init__(self, *args, **kwargs):
        self._rsa = RSA(*args, **kwargs)

    def __call__(self, assembly):
        rsa = self._rsa(assembly)
        return 1 - rsa


class RDMSimilarity(object):
    def __init__(self, comparison_coord=XarrayDefaults.stimulus_coord):
        self._comparison_coord = comparison_coord

    def __call__(self, rdm_assembly1, rdm_assembly2):
        assert all(rdm_assembly1[self._comparison_coord].values == rdm_assembly2[self._comparison_coord].values)
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
