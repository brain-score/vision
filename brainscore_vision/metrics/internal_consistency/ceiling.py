import scipy.stats
import xarray as xr
from result_caching import store
from tqdm import tqdm

from brainio.assemblies import walk_coords, DataAssembly
from brainscore_core import Metric, Score
from brainscore_vision.metric_helpers import Defaults as XarrayDefaults
from brainscore_vision.metric_helpers.transformations import CrossValidationSingle
from brainscore_vision.metric_helpers.xarray_utils import XarrayCorrelation
from brainscore_vision.metrics import Ceiling


class NoCeiling(Ceiling):
    def __call__(self, assembly: DataAssembly) -> Score:
        return Score(1)


class _SplitHalvesConsistency(Ceiling):
    """
    Computes the consistency within an assembly with repetitions,
    by splitting the assembly in halves and computing consistency between the two halves.
    """

    class Defaults:
        split_coord = 'repetition'

    def __init__(self, consistency_metric, aggregate=None,
                 split_coord=Defaults.split_coord, cross_validation_kwargs=None):
        correction = SpearmanBrownCorrection()
        self._consistency = self.SplitHalfWrapper(split_coord=split_coord,
                                                  consistency_metric=consistency_metric, correction=correction)
        self._aggregate = aggregate
        cross_validation_defaults = dict(train_size=0.5, split_coord=split_coord,
                                         stratification_coord=None, unique_split_values=True)
        cross_validation_kwargs = {**cross_validation_defaults, **(cross_validation_kwargs or {})}
        self._cross_validation = CrossValidationSingle(**cross_validation_kwargs)

    def __call__(self, assembly):
        return self._cross_validation(assembly, apply=self._consistency, aggregate=self._aggregate)

    class SplitHalfWrapper:
        def __init__(self, split_coord, consistency_metric: Metric, correction):
            self._split_coord = split_coord
            self._consistency_metric = consistency_metric
            self._correction = correction

        def __call__(self, half1, half2):
            half1, half2 = self._average_repetitions(half1), self._average_repetitions(half2)
            consistency = self._consistency_metric(half1, half2)
            corrected_consistency = self._correction(consistency, n=2)
            return corrected_consistency

        def _average_repetitions(self, assembly):
            repetition_dims = assembly[self._split_coord].dims
            nonrepetition_coords = [coord for coord, dims, values in walk_coords(assembly)
                                    if dims == repetition_dims and coord != self._split_coord]
            average = assembly.multi_groupby(nonrepetition_coords).mean(dim=repetition_dims)
            return average


class PearsonCorrelation:
    """
    Computes the Pearson r between two halves of an assembly.
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


class InternalConsistency(Ceiling):
    def __init__(self,
                 consistency_metric: Metric = None,
                 aggregate=None,
                 split_coord=_SplitHalvesConsistency.Defaults.split_coord):
        """
        Creates a class to estimate the ceiling for a given assembly, based on split-half consistencies.

        :param consistency_metric: The metric to compare two halves. :class:`~.PearsonCorrelation` by default.
        :param aggregate: An optional function to aggregate the output of the `consistency_metric`.
        :param split_coord: Over which coordinate to split to obtain the two halves.
        """
        if not consistency_metric:
            consistency_metric = PearsonCorrelation(
                stimulus_coord=XarrayDefaults.stimulus_coord, neuroid_dim=XarrayDefaults.neuroid_dim,
                neuroid_coord=XarrayDefaults.neuroid_coord)
            aggregate = consistency_metric.aggregate
        self._consistency = _SplitHalvesConsistency(consistency_metric=consistency_metric,
                                                    aggregate=aggregate,
                                                    split_coord=split_coord)

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


class NeuronalPropertyCeiling:
    def __init__(self, similarity_metric: Metric):
        self.similarity_metric = similarity_metric

    def __call__(self, assembly):
        self.assembly = assembly
        return self._ceiling(self.similarity_metric.property_name)

    @store()
    def _ceiling(self, identifier):
        return self.similarity_metric(self.assembly, self.assembly)
