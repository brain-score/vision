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

    def __call__(self, assembly1, assembly2, rdm_dim='presentation'):
        """
        :param mkgu.assemblies.NeuroidAssembly assembly1:
        :param mkgu.assemblies.NeuroidAssembly assembly2:
        :param Union[str, list] rdm_dim: indicate the dimension along which the RDM/RSA was computed,
            either with a string for a repeated dimension or with a list for two different dimension names
        :return: mkgu.assemblies.DataAssembly
        """
        # get upper triangulars
        assert isinstance(rdm_dim, str) or len(rdm_dim) == 2
        joint_dim = '{}-{}'.format(*rdm_dim if not isinstance(rdm_dim, str) else (rdm_dim, rdm_dim))
        triu1 = self._preprocess_assembly(assembly1, rdm_dim=rdm_dim, joint_dim=joint_dim)
        triu2 = self._preprocess_assembly(assembly2, rdm_dim=rdm_dim, joint_dim=joint_dim)
        assert triu1.dims == triu2.dims
        assert triu1.shape == triu2.shape
        joint_dim_index = np.where(np.array(triu1.dims) == joint_dim)[0][0]

        # compute correlations
        def insert_rdm_slice(combination):
            combination.insert(joint_dim_index, slice(None))
            return combination

        indices = [insert_rdm_slice(list(combination)) for combination in
                   itertools.product(*[list(range(len(triu1[dim])))
                                       for dim in filter(lambda dim: dim != joint_dim, triu1.dims)])]

        def _corr(vals1, vals2):
            corr = np.corrcoef(vals1, vals2)
            assert len(corr.shape) == 2 and corr.shape[0] == 2 and corr.shape[1] == 2
            np.testing.assert_almost_equal(np.diag(corr), [1, 1])
            return corr[0, 1]

        corrs = np.array([_corr(triu1.values[ids], triu2.values[ids]) for ids in indices])
        corrs = corrs.reshape(triu1.shape[:joint_dim_index] + triu1.shape[joint_dim_index+1:])

        # package in assembly
        coords = {coord: values for coord, values in triu1.coords.items() if coord is not joint_dim}
        dims = list(triu1.dims)
        dims.remove(joint_dim)
        return DataAssembly(corrs, coords=coords, dims=dims)

    def _preprocess_assembly(self, assembly, rdm_dim, joint_dim):
        rdm_dim_indices, = np.where(np.array(assembly.dims) == rdm_dim)
        adjacent_dims = list(filter(lambda dim: dim != rdm_dim, assembly.dims))
        assert len(rdm_dim_indices) == 2
        self._assert_diagonal_zero(assembly, rdm_dim_indices, diag_dimensions=list(adjacent_dims) + [rdm_dim])

        # get upper triangulars
        triangular_indices = np.triu_indices(assembly[rdm_dim].shape[0], k=1)
        indices = [slice(None) if i not in rdm_dim_indices else triangular_indices[np.where(rdm_dim_indices == i)[0][0]]
                   for i in range(len(assembly.dims))]
        triu = assembly.values[indices]

        # package in assembly again
        coords = {coord: assembly[coord] for coord in assembly.coords if coord != rdm_dim}
        coords[joint_dim] = ['{}-{}'.format(*assembly[rdm_dim][[i1, i2]].values)
                             # ^ hack around xarray not allowing 2D coords
                             for i1, i2 in zip(*triangular_indices)]
        triu_dims = list(assembly.dims)
        triu_dims.remove(rdm_dim)
        triu_dims = [dim if dim != rdm_dim else joint_dim for dim in triu_dims]
        return DataAssembly(triu, coords=coords, dims=triu_dims)

    def _assert_diagonal_zero(self, assembly, rdm_dim_indices, diag_dimensions):
        assert all(len(assembly[dim].shape) == 1 for dim in diag_dimensions)
        zero_target = np.zeros([assembly[dim].shape[0] for dim in diag_dimensions])
        diag_values = np.diagonal(assembly, axis1=rdm_dim_indices[0], axis2=rdm_dim_indices[1])
        assert np.allclose(diag_values, zero_target, atol=1e-10)
