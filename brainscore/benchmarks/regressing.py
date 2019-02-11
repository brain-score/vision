from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.loaders import DicarloMajaj2015Loader, MovshonFreemanZiemba2013Loader
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation


class DicarloMajaj2015Region(BenchmarkBase):
    # We here treat V4 and IT as separate benchmarks
    # even though they were collected from the same brains in the same sessions.
    def __init__(self, region):
        self.region = region
        loader = DicarloMajaj2015Loader()
        assembly_repetition = loader(average_repetition=False)
        assembly_repetition = assembly_repetition.sel(region=region)
        self._assembly = loader.average_repetition(assembly_repetition)

        self._similarity_metric = CrossRegressedCorrelation()
        name = f'dicarlo.Majaj2015.{region}-regressing'
        ceiler = InternalConsistency()
        super(DicarloMajaj2015Region, self).__init__(
            name=name, ceiling_func=lambda: ceiler(assembly_repetition))

    def __call__(self, candidate):
        candidate.start_recording(self.region)
        source_assembly = candidate.look_at(self._assembly.stimulus_set)
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        return ceil_score(raw_score, self.ceiling)


class DicarloMajaj2015V4(DicarloMajaj2015Region):
    def __init__(self):
        super(DicarloMajaj2015V4, self).__init__(region='V4')


class DicarloMajaj2015IT(DicarloMajaj2015Region):
    def __init__(self):
        super(DicarloMajaj2015IT, self).__init__(region='IT')


class MovshonFreemanZiemba2013Region(BenchmarkBase):
    def __init__(self, region):
        self.region = region
        loader = MovshonFreemanZiemba2013Loader()
        assembly_repetition = loader(average_repetition=False)
        assembly_repetition = assembly_repetition.sel(region=region).stack(neuroid=['neuroid_id'])
        self._assembly = loader.average_repetition(assembly_repetition)
        self._assembly.stimulus_set.name = self._assembly.stimulus_set_name

        self._similarity_metric = CrossRegressedCorrelation()
        name = f'movshon.FreemanZiemba2013.{region}-regressing'
        ceiler = InternalConsistency()
        super(MovshonFreemanZiemba2013Region, self).__init__(
            name=name, ceiling_func=lambda: ceiler(assembly_repetition))

    def __call__(self, candidate):
        candidate.record_from(area=self.region)
        source_assembly = candidate.look_at(self._assembly.stimulus_set)
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        return ceil_score(raw_score, self.ceiling)


class MovshonFreemanZiemba2013V1(MovshonFreemanZiemba2013Region):
    def __init__(self):
        super(MovshonFreemanZiemba2013V1, self).__init__(region='V1')


class MovshonFreemanZiemba2013V2(MovshonFreemanZiemba2013Region):
    def __init__(self):
        super(MovshonFreemanZiemba2013V2, self).__init__(region='V2')
