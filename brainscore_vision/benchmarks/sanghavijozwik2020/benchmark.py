import numpy as np

import brainscore_vision
from brainscore_vision.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore_vision.metrics.ceiling import InternalConsistency
from brainscore_vision.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore_vision.utils import LazyLoad


BIBTEX = """"""


def DicarloSanghaviJozwik2020V4PLS():
    return _DicarloSanghaviJozwik2020Region('V4')


def DicarloSanghaviJozwik2020ITPLS():
    return _DicarloSanghaviJozwik2020Region('IT')


class _DicarloSanghaviJozwik2020Region(NeuralBenchmark):
    def __init__(self, region: str):
        self._assembly_repetition = LazyLoad(lambda region=region: self._load_assembly(region, average_repetitions=False))
        self._assembly = LazyLoad(lambda region=region: self._load_assembly(region, average_repetitions=True))
        self._identifier_metric_suffix = 'pls'
        self._similarity_metric = CrossRegressedCorrelation(
            regression=pls_regression(), correlation=pearsonr_correlation(),
            crossvalidation_kwargs=dict(stratification_coord=None))
        self._ceiler = InternalConsistency()
        self._visual_degrees = 8
        self._number_of_trials = 37
        super(_DicarloSanghaviJozwik2020Region, self).__init__(
            identifier=f'dicarlo.SanghaviJozwik2020.{region}-{self._identifier_metric_suffix}', version=1,
            assembly=self._assembly, similarity_metric=self._similarity_metric,
            visual_degrees=self._visual_degrees, number_of_trials=self._number_of_trials,
            ceiling_func=lambda: self._ceiler(self._assembly_repetition),
            parent=region,
            bibtex=BIBTEX
        )

    def _load_assembly(self, region, average_repetitions):
        # TODO: load data set must be created within brainscore_vision
        assembly = brainscore_vision.load_dataset('dicarlo.SanghaviJozwik2020')

        assembly = assembly.sel(region=region)
        assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
        assembly.load()
        assembly = assembly.sel(time_bin_id=0)  # 70-170ms
        assembly = assembly.squeeze('time_bin')
        assert self._number_of_trials == len(np.unique(assembly.coords['repetition']))
        assert self._visual_degrees == assembly.attrs['image_size_degree']
        if average_repetitions:
            assembly = average_repetition(assembly)
        return assembly
