import scipy.stats
import xarray as xr
from tqdm import tqdm

from brainio.assemblies import walk_coords
from brainscore.metrics import Score
from brainscore.metrics.rdm import RDMMetric
from brainscore.metrics.cka import CKAMetric
from brainscore.metrics.transformations import CrossValidationSingle
from brainscore.metrics.xarray_utils import Defaults as XarrayDefaults
from brainscore.metrics.xarray_utils import XarrayCorrelation
from result_caching import store
import random
import numpy as np


class Ceiling:
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


class CKAConsistency(Ceiling):
    def __init__(self):
        cka = CKAMetric()
        self._consistency = _SplitHalvesConsistency(consistency=cka)

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
    def __init__(self, similarity_metric):
        self.similarity_metric = similarity_metric

    def __call__(self, assembly):
        self.assembly = assembly
        return self._ceiling(self.similarity_metric.property_name)

    @store()
    def _ceiling(self, identifier):
        return self.similarity_metric(self.assembly, self.assembly)

ceilings = {
    'cons': InternalConsistency,
    None: NoCeiling,
}


class SplitHalvesConsistencyBaker:
    # following
    # https://github.com/brain-score/brain-score/blob/c51b8aa2c94212a9ac56c06c556afad0bb0a3521/brainscore/metrics/ceiling.py#L25-L96

    def __init__(self, num_splits: int, split_coordinate: str, consistency_metric, image_types):
        """
        :param num_splits: how many times to create two halves
        :param split_coordinate: over which coordinate to split the assembly into halves
        :param consistency_metric: which metric to use to compute the consistency of two halves
        """
        self.num_splits = num_splits
        self.split_coordinate = split_coordinate
        self.consistency_metric = consistency_metric
        self.image_types = image_types

    def __call__(self, assembly) -> Score:
        consistencies, uncorrected_consistencies = [], []
        splits = range(self.num_splits)
        for _ in splits:
            num_subjects = len(set(assembly["subject"].values))
            half1_subjects = random.sample(range(1, num_subjects), (num_subjects // 2))
            half1 = assembly[
                {'presentation': [subject in half1_subjects for subject in assembly['subject'].values]}]
            half2 = assembly[
                {'presentation': [subject not in half1_subjects for subject in assembly['subject'].values]}]
            consistency = self.consistency_metric(half1, half2, image_types=self.image_types, isCeiling=True)
            uncorrected_consistencies.append(consistency)
            # Spearman-Brown correction for sub-sampling
            corrected_consistency = 2 * consistency / (1 + (2 - 1) * consistency)
            consistencies.append(corrected_consistency)
        consistencies = Score(consistencies, coords={'split': splits}, dims=['split'])
        uncorrected_consistencies = Score(uncorrected_consistencies, coords={'split': splits}, dims=['split'])
        average_consistency = consistencies.median('split')
        average_consistency.attrs['raw'] = consistencies
        average_consistency.attrs['uncorrected_consistencies'] = uncorrected_consistencies
        ceiling_error = np.std(consistencies)
        return average_consistency, ceiling_error
