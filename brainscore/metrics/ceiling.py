import scipy.stats

from brainscore.assemblies import DataAssembly, walk_coords
from brainscore.metrics import build_score
from brainscore.metrics.transformations import Transformations, CrossValidationSingle
from brainscore.metrics.xarray_utils import Defaults as XarrayDefaults
from brainscore.metrics.xarray_utils import XarrayCorrelation


class Ceiling(object):
    def __call__(self, assembly):
        raise NotImplementedError()


class NoCeiling(Ceiling):
    def __call__(self, assembly):
        return build_score(None, center=DataAssembly(1), error=DataAssembly(0))


class InternalConsistency(Ceiling):
    """
    Computes the consistency within an assembly with repetitions.
    """

    class Defaults:
        split_coord = 'repetition'

    def __init__(self, split_coord=Defaults.split_coord, stimulus_coord=XarrayDefaults.stimulus_coord,
                 neuroid_dim=XarrayDefaults.neuroid_dim, neuroid_coord=XarrayDefaults.neuroid_coord):
        consistency = SplitHalfConsistency(stimulus_coord=stimulus_coord, neuroid_dim=neuroid_dim,
                                           neuroid_coord=neuroid_coord)
        self._consistency = self.SplitHalfWrapper(split_coord=split_coord, consistency=consistency)
        cross_validation = CrossValidationSingle(train_size=0.5, split_coord=split_coord)
        self._transformations = Transformations([cross_validation])

    def __call__(self, assembly):
        return self._transformations(assembly, metric=self._consistency)

    class SplitHalfWrapper:
        def __init__(self, split_coord, consistency):
            self._split_coord = split_coord
            self._consistency = consistency

        def __call__(self, half1, half2):
            half1, half2 = self._average_repetitions(half1), self._average_repetitions(half2)
            return self._consistency(half1, half2)

        def _average_repetitions(self, assembly):
            repetition_dims = assembly[self._split_coord].dims
            nonrepetition_coords = [coord for coord, dims, values in walk_coords(assembly)
                                    if dims == repetition_dims and coord != self._split_coord]
            average = assembly.multi_groupby(nonrepetition_coords).mean(dim=repetition_dims)
            return average

        def aggregate(self, scores):
            return self._consistency.aggregate(scores)


class SplitHalfConsistency:
    """
    Computes the consistency between two halves of an assembly.
    """

    def __init__(self, stimulus_coord=XarrayDefaults.stimulus_coord,
                 neuroid_dim=XarrayDefaults.neuroid_dim, neuroid_coord=XarrayDefaults.neuroid_coord):
        correlation = scipy.stats.pearsonr
        self._correlation = XarrayCorrelation(correlation, stimulus_coord=stimulus_coord, neuroid_coord=neuroid_coord)
        self._neuroid_dim = neuroid_dim

    def __call__(self, half1, half2):
        return self._correlation(half1, half2)

    def aggregate(self, scores):
        return scores.median(dim=self._neuroid_dim)


def spearman_brown_correction(correlation, n):
    return n * correlation / (1 + (n - 1) * correlation)


ceilings = {
    'cons': InternalConsistency,
    None: NoCeiling,
}
