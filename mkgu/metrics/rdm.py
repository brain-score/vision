import itertools

import numpy as np

from mkgu.assemblies import DataAssembly
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


class RDMCorrelationCoefficient(Similarity):
    """
    Computes a coefficient for the similarity between two `RDM`s, using the upper triangular regions

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __init__(self):
        self._rdm_coord = 'presentation'

    def __call__(self, assembly1, assembly2):
        triu1, triu2 = self._preprocess_assembly(assembly1), self._preprocess_assembly(assembly2)
        assert triu1.dims == triu2.dims
        assert triu1.shape == triu2.shape
        indices = [[slice(None)] + list(combination) for combination in
                   itertools.product(*[list(range(len(triu1[dim]))) for dim in triu1.dims[1:]])]

        def _corr(vals1, vals2):
            corr = np.corrcoef(vals1, vals2)
            assert len(corr.shape) == 2 and corr.shape[0] == 2 and corr.shape[1] == 2
            np.testing.assert_almost_equal(np.diag(corr), [1, 1])
            return corr[0, 1]

        corrs = np.array([_corr(triu1.values[ids], triu2.values[ids]) for ids in indices]).reshape(triu1.shape[1:])
        coords = {coord: values for coord, values in triu1.coords.items() if coord is not self._rdm_coord}
        return DataAssembly(corrs, coords=coords, dims=triu1.dims[1:])

    def _preprocess_assembly(self, assembly):
        rdm_coord_indices, = np.where(np.array(assembly.dims) == self._rdm_coord)
        adjacent_dims = set(assembly.dims) - {self._rdm_coord}

        assert len(rdm_coord_indices) == 2
        self._assert_diagonal_zero(assembly, rdm_coord_indices, adjacent_dims)

        # get triangulars
        triangular_indices = np.triu_indices(assembly[self._rdm_coord].shape[0], k=1)
        indices = [slice(None) if i not in rdm_coord_indices else triangular_indices[rdm_coord_indices[i]]
                   for i in range(len(assembly.dims))]
        triu = assembly.values[indices]

        # package in assembly again
        coords = {coord: assembly[coord] for coord in assembly.coords if coord != self._rdm_coord}
        joint_rdm_coord = self._rdm_coord
        coords[joint_rdm_coord] = ['{}-{}'.format(*assembly[self._rdm_coord][[i1, i2]].values)
                                   # ^ hack around xarray not allowing 2D coords
                                   for i1, i2 in zip(*triangular_indices)]
        triu_dims = [joint_rdm_coord] + list(adjacent_dims)
        return DataAssembly(triu, coords=coords, dims=triu_dims)

    def _assert_diagonal_zero(self, assembly, rdm_coord_indices, adjacent_dims):
        diag_dimensions = list(adjacent_dims) + [self._rdm_coord]
        assert all(len(assembly[dim].shape) == 1 for dim in diag_dimensions)
        zero_target = np.zeros([assembly[dim].shape[0] for dim in diag_dimensions])
        diag_values = np.diagonal(assembly, axis1=rdm_coord_indices[0], axis2=rdm_coord_indices[1])
        assert np.allclose(diag_values, zero_target, atol=1e-10)
