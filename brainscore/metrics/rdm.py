import numpy as np
from scipy.stats import spearmanr

from brainio.assemblies import DataAssembly, walk_coords
from brainscore.metrics import Score
from brainscore.metrics.transformations import TestOnlyCrossValidation
from brainscore.metrics.xarray_utils import Defaults as XarrayDefaults


class RDMCrossValidated:
    """
    Computes a coefficient for the similarity between two `RDM`s, using the upper triangular regions

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __init__(self, representation_dim=XarrayDefaults.neuroid_dim, comparison_coord=XarrayDefaults.stimulus_coord,
                 crossvalidation_kwargs=None):
        self._metric = RDMMetric(representation_dim=representation_dim, comparison_coord=comparison_coord)
        crossvalidation_defaults = dict(test_size=.9)  # leave 10% out
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}
        self._cross_validation = TestOnlyCrossValidation(**crossvalidation_kwargs)

    def __call__(self, assembly1, assembly2):
        """
        :param brainscore.assemblies.NeuroidAssembly assembly1:
        :param brainscore.assemblies.NeuroidAssembly assembly2:
        :return: brainscore.assemblies.DataAssembly
        """

        return self._cross_validation(assembly1, assembly2, apply=self._metric)


class RDMMetric:
    """
    Computes a coefficient for the similarity between two `RDM`s, using the upper triangular regions

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __init__(self,
                 representation_dim: str = XarrayDefaults.neuroid_dim,
                 comparison_coord: str = XarrayDefaults.stimulus_coord,
                 similarity_function=None):
        self._representation_dim = representation_dim
        self._rdm = RDM(representation_dim=representation_dim, similarity_function=similarity_function)
        self._similarity = RDMSimilarity(comparison_coord=comparison_coord)

    def __call__(self, assembly1: DataAssembly, assembly2: DataAssembly) -> Score:
        rdm1 = self._rdm(assembly1)
        rdm2 = self._rdm(assembly2)
        similarity = self._similarity(rdm1, rdm2)
        return Score(similarity)


class RDM:
    """
    Representational Dissimilarity Matrix.
    Converts an assembly of `presentation x neuroid` into a `neuroid x neuroid` RDM.

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __init__(self, *args, **kwargs):
        self._rsa = RSA(*args, **kwargs)

    def __call__(self, assembly: DataAssembly) -> DataAssembly:
        rsa = self._rsa(assembly)
        return 1 - rsa


class RSA:
    """
    Representational Similarity Analysis.
    Converts an assembly of `presentation x neuroid` into a `neuroid x neuroid` RSA matrix.

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """

    def __init__(self, representation_dim: str = XarrayDefaults.neuroid_dim, similarity_function=None):
        self._representation_dim = representation_dim
        self._similarity_function = similarity_function or np.corrcoef

    def __call__(self, assembly: DataAssembly) -> DataAssembly:
        assert len(assembly.dims) == 2
        assembly = assembly.transpose(..., self._representation_dim)  # ensure last dim is self._representation_dim
        similarities = self._similarity_function(assembly)
        coords = {coord: coord_value for coord, coord_value in assembly.coords.items() if
                  coord != self._representation_dim}
        dims = [dim if dim != self._representation_dim else assembly.dims[(i - 1) % len(assembly.dims)]
                for i, dim in enumerate(assembly.dims)]
        return DataAssembly(similarities, coords=coords, dims=dims)


class RDMSimilarity:
    def __init__(self, comparison_coord: str = XarrayDefaults.stimulus_coord):
        self._comparison_coord = comparison_coord

    def __call__(self, rdm_assembly1: DataAssembly, rdm_assembly2: DataAssembly) -> float:
        # align
        rdm_assembly1 = self.multishape_preserved_sort(rdm_assembly1)
        rdm_assembly2 = self.multishape_preserved_sort(rdm_assembly2)
        assert (rdm_assembly1[self._comparison_coord].values == rdm_assembly2[self._comparison_coord].values).all()

        triu1 = self._triangulars(rdm_assembly1.values)
        triu2 = self._triangulars(rdm_assembly2.values)
        corr, p = spearmanr(triu1, triu2)
        return corr

    def _triangulars(self, values):
        assert len(values.shape) == 2 and values.shape[0] == values.shape[1]
        # ensure diagonal is zero
        diag = np.diag(values)
        diag = np.nan_to_num(diag, nan=0, copy=True)  # we also accept nans in the diagonal from correlating zeros
        np.testing.assert_almost_equal(diag, 0)
        # index and retrieve upper triangular
        triangular_indices = np.triu_indices(values.shape[0], k=1)
        return values[triangular_indices]

    def multishape_preserved_sort(self, assembly):
        comparison_dims = assembly[self._comparison_coord].dims
        assert set(assembly.dims) == set(comparison_dims), "multi-dimensional case not implemented"
        indices = np.argsort(assembly[self._comparison_coord].values)
        assembly = type(assembly)(assembly.values[np.ix_(indices, indices)],
                                  coords={coord: (dims, values[indices] if dims == comparison_dims else values)
                                          for coord, dims, values in walk_coords(assembly)},
                                  dims=assembly.dims)
        return assembly
