import scipy.stats
import xarray as xr
from tqdm import tqdm

from brainio_core.assemblies import walk_coords
from brainscore.metrics import Score
from brainscore.metrics.rdm import RDMMetric
from brainscore.metrics.transformations import CrossValidationSingle
from brainscore.metrics.xarray_utils import Defaults as XarrayDefaults
from brainscore.metrics.xarray_utils import XarrayCorrelation


class Ceiling(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class NoCeiling(Ceiling):
    def __call__(self):
        return Score(1)


class _SplitHalvesConsistency(Ceiling):
    """
    Computes the consistency within an assembly with repetitions,
    by splitting the assembly in halves and computing consistency between the two halves.
    """

    class Defaults:
        split_coord = 'repetition'

    def __init__(self, consistency, split_coord=Defaults.split_coord, cross_validation_kwargs=None, aggregate=None):
        correction = SpearmanBrownCorrection()
        self._consistency = self.SplitHalfWrapper(split_coord=split_coord,
                                                  consistency=consistency, correction=correction)
        self._aggregate = aggregate
        cross_validation_defaults = dict(train_size=0.5, split_coord=split_coord,
                                         stratification_coord=None, unique_split_values=True)
        cross_validation_kwargs = {**cross_validation_defaults, **(cross_validation_kwargs or {})}
        self._cross_validation = CrossValidationSingle(**cross_validation_kwargs)

    def __call__(self, assembly):
        return self._cross_validation(assembly, apply=self._consistency, aggregate=self._aggregate)

    class SplitHalfWrapper:
        def __init__(self, split_coord, consistency, correction):
            self._split_coord = split_coord
            self._consistency = consistency
            self._correction = correction

        def __call__(self, half1, half2):
            half1, half2 = self._average_repetitions(half1), self._average_repetitions(half2)
            consistency = self._consistency(half1, half2)
            consistency = self._correction(consistency, n=2)
            return consistency

        def _average_repetitions(self, assembly):
            repetition_dims = assembly[self._split_coord].dims
            nonrepetition_coords = [coord for coord, dims, values in walk_coords(assembly)
                                    if dims == repetition_dims and coord != self._split_coord]
            average = assembly.multi_groupby(nonrepetition_coords).mean(dim=repetition_dims)
            return average


class InternalConsistency(Ceiling):
    def __init__(self,
                 split_coord=_SplitHalvesConsistency.Defaults.split_coord, stimulus_coord=XarrayDefaults.stimulus_coord,
                 neuroid_dim=XarrayDefaults.neuroid_dim, neuroid_coord=XarrayDefaults.neuroid_coord):
        consistency = SplitHalfConsistency(stimulus_coord=stimulus_coord, neuroid_dim=neuroid_dim,
                                           neuroid_coord=neuroid_coord)
        self._consistency = _SplitHalvesConsistency(consistency=consistency, split_coord=split_coord,
                                                    aggregate=consistency.aggregate)

    def __call__(self, assembly):
        return self._consistency(assembly)


class SplitHalfConsistency:
    """
    Computes the consistency between two halves of an assembly.
    """

    def __init__(self, stimulus_coord=XarrayDefaults.stimulus_coord,
                 neuroid_dim=XarrayDefaults.neuroid_dim, neuroid_coord=XarrayDefaults.neuroid_coord):
        correlation = scipy.stats.pearsonr
        self._correlation = XarrayCorrelation(correlation, correlation_coord=stimulus_coord,
                                              neuroid_coord=neuroid_coord)
        self._neuroid_dim = neuroid_dim

    def __call__(self, half1, half2):
        return self._correlation(half1, half2)

    def aggregate(self, scores):
        return scores.median(dim=self._neuroid_dim)


class RDMConsistency(Ceiling):
    def __init__(self):
        rdm = RDMMetric()
        self._consistency = _SplitHalvesConsistency(consistency=rdm)

    def __call__(self, assembly):
        return self._consistency(assembly)


class SpearmanBrownCorrection:
    """
    Applies Spearman-Brown correction to all passed values.
    """

    def __call__(self, correlations, n):
        return xr.apply_ufunc(lambda correlation: self.correct(correlation, n), correlations)

    def correct(self, correlation, n):
        return n * correlation / (1 + (n - 1) * correlation)


class TemporalCeiling:
    def __init__(self, ceiling):
        """
        :param ceiling: the ceiling to use per time-bin
        """
        self.ceiling = ceiling

    def __call__(self, assembly):
        ceilings = []
        for time_bin in tqdm(assembly['time_bin'].values, desc='time-ceiling'):
            ceiling = self.ceiling(assembly.sel(time_bin=time_bin))
            ceiling = ceiling.expand_dims('time_bin')
            ceiling['time_bin'] = [str(time_bin)]
            ceilings.append(ceiling)
        ceiling = Score.merge(*ceilings)
        return ceiling


ceilings = {
    'cons': InternalConsistency,
    None: NoCeiling,
}
