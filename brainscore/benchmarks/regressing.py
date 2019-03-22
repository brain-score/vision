import numpy as np

from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.loaders import assembly_loaders
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation


class NeuralBenchmark(BenchmarkBase):
    def __init__(self, identifier, assembly, similarity_metric, ceiling_func):
        super(NeuralBenchmark, self).__init__(identifier=identifier, ceiling_func=ceiling_func)
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        region = np.unique(self._assembly['region'])
        assert len(region) == 1
        self.region = region[0]

    def __call__(self, candidate):
        candidate.start_recording(self.region)
        source_assembly = candidate.look_at(self._assembly.stimulus_set)
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        return ceil_score(raw_score, self.ceiling)


def _DicarloMajaj2015Region(region):
    loader = assembly_loaders[f'dicarlo.Majaj2015.highvar.{region}']
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)

    similarity_metric = CrossRegressedCorrelation()
    identifier = f'dicarlo.Majaj2015.{region}-regressing'
    ceiler = InternalConsistency()
    return NeuralBenchmark(identifier=identifier, assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition))


def DicarloMajaj2015V4():
    return _DicarloMajaj2015Region('V4')


def DicarloMajaj2015IT():
    return _DicarloMajaj2015Region('IT')


def _MovshonFreemanZiemba2013Region(region):
    loader = assembly_loaders[f'movshon.FreemanZiemba2013.{region}']
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation()
    identifier = f'movshon.FreemanZiemba2013.{region}-regressing'
    ceiler = InternalConsistency()
    return NeuralBenchmark(identifier=identifier, assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition))


def MovshonFreemanZiemba2013V1():
    return _MovshonFreemanZiemba2013Region('V1')


def MovshonFreemanZiemba2013V2():
    return _MovshonFreemanZiemba2013Region('V2')
