import logging
from abc import ABC
from collections import OrderedDict

import numpy as np
from numpy.random import RandomState
from result_caching import cache, store
from xarray import DataArray

from brainscore.benchmarks.loaders import load_assembly, DicarloMajaj2015Loader, ToliasCadena2017Loader
from brainscore.metrics import Score
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.correlation import CrossCorrelation
from brainscore.metrics.transformations import subset
from brainscore.utils import fullname, LazyLoad


class Benchmark(ABC):
    def __call__(self, source_assembly):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def ceiling(self):
        raise NotImplementedError()


class BenchmarkBase(Benchmark):
    def __init__(self, name, assembly, similarity_metric, ceiling_func):
        self._name = name
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        self._ceiling_func = ceiling_func
        self._logger = logging.getLogger(fullname(self))

    @property
    def name(self):
        return self._name

    @property
    def assembly(self):
        return self._assembly

    def __call__(self, source_assembly):
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        ceiled_score = self._ceil_score(raw_score)
        ceiled_score.attrs[Score.RAW_VALUES_KEY] = raw_score
        return ceiled_score

    def _ceil_score(self, raw_score):
        return raw_score / self.ceiling

    @property
    @store()
    def ceiling(self):
        return self._ceiling_func()


class DicarloMajaj2015Region(BenchmarkBase):
    # We here treat V4 and IT as separate benchmarks
    # even though they were collected from the same brains in the same sessions.
    def __init__(self, region):
        loader = DicarloMajaj2015Loader()
        assembly_repetitions = loader(average_repetition=False)
        assembly_repetitions = assembly_repetitions.sel(region=region)
        assembly_repetitions = split_assembly(assembly_repetitions,
                                              named_ratios=[('train', .5), ('validation', .3), ('test', .2)])
        training_assembly_repetitions, validation_assembly_repetitions, test_assembly_repetitions = assembly_repetitions
        assemblies = [loader.average_repetition(assembly_repetition) for assembly_repetition in assembly_repetitions]
        training_assembly, validation_assembly, test_assembly = assemblies

        similarity_metric = CrossCorrelation()
        name = f'dicarlo.Majaj2015.{region}'
        ceiler = InternalConsistency()
        super(DicarloMajaj2015Region, self).__init__(
            name=f"{name}-test", assembly=test_assembly,
            similarity_metric=similarity_metric, ceiling_func=lambda: ceiler(test_assembly_repetitions))
        self.training_benchmark = BenchmarkBase(
            name=f"{name}-training", assembly=training_assembly,
            similarity_metric=similarity_metric, ceiling_func=lambda: ceiler(training_assembly_repetitions))
        self.validation_benchmark = BenchmarkBase(
            name=f"{name}-validation", assembly=validation_assembly,
            similarity_metric=similarity_metric, ceiling_func=lambda: ceiler(validation_assembly_repetitions))


class DicarloMajaj2015V4(DicarloMajaj2015Region):
    def __init__(self):
        super(DicarloMajaj2015V4, self).__init__(region='V4')


class DicarloMajaj2015IT(DicarloMajaj2015Region):
    def __init__(self):
        super(DicarloMajaj2015IT, self).__init__(region='IT')


class ToliasCadena2017(BenchmarkBase):
    def __init__(self):
        loader = ToliasCadena2017Loader()
        assembly_repetitions = loader(average_repetition=False)
        assembly_repetitions = split_assembly(assembly_repetitions,
                                              named_ratios=[('train', .5), ('validation', .3), ('test', .2)])
        training_assembly_repetitions, validation_assembly_repetitions, test_assembly_repetitions = assembly_repetitions
        assemblies = [loader.average_repetition(assembly_repetition) for assembly_repetition in assembly_repetitions]
        training_assembly, validation_assembly, test_assembly = assemblies

        similarity_metric = CrossCorrelation()
        name = 'tolias.Cadena2017'
        ceiler = InternalConsistency(split_coord='repetition_id')
        super(ToliasCadena2017, self).__init__(
            name=f"{name}-test", assembly=test_assembly,
            similarity_metric=similarity_metric, ceiling_func=lambda: ceiler(test_assembly_repetitions))
        self.training_benchmark = BenchmarkBase(
            name=f"{name}-training", assembly=training_assembly,
            similarity_metric=similarity_metric, ceiling_func=lambda: ceiler(training_assembly_repetitions))
        self.validation_benchmark = BenchmarkBase(
            name=f"{name}-validation", assembly=validation_assembly,
            similarity_metric=similarity_metric, ceiling_func=lambda: ceiler(validation_assembly_repetitions))


benchmark_pool = {
    'dicarlo.Majaj2015.V4': LazyLoad(lambda: DicarloMajaj2015V4()),
    'dicarlo.Majaj2015.IT': LazyLoad(lambda: DicarloMajaj2015IT()),
    'tolias.Cadena2017': LazyLoad(lambda: ToliasCadena2017()),
}


@cache()
def load(name):
    if name not in benchmark_pool:
        raise ValueError("Unknown benchmark '{}' - must choose from {}".format(name, list(benchmark_pool.keys())))
    return benchmark_pool[name]


def split_assembly(assembly, on='image_id', named_ratios=(('map', .8), ('test', .2))):
    # TODO: this method should be in packaging
    named_ratios = OrderedDict(named_ratios)
    dim = assembly[on].dims[0]
    rng = RandomState(seed=1)
    choice_options = unique_preserved_order(assembly[on].values)
    num_choices = {name: np.round(ratio * len(choice_options)).astype(int) for name, ratio in named_ratios.items()}
    assert sum(num_choices.values()) == len(choice_options)

    assemblies = []
    for name, num_choice in num_choices.items():
        choice = rng.choice(choice_options, size=num_choice, replace=False)
        choice_options = sorted(set(choice_options) - set(choice))  # sort to avoid introducing set diff randomness
        subset_indexer = DataArray(np.zeros(len(choice)), coords={on: choice}, dims=[on]).stack(**{dim: [on]})
        choice_assembly = subset(assembly, subset_indexer, dims_must_match=False)  # 0 exactly identical for two runs

        choice_assembly.attrs['stimulus_set'] = assembly.stimulus_set[
            assembly.stimulus_set['image_id'].isin(choice_assembly['image_id'].values)]
        choice_assembly.stimulus_set.name = assembly.stimulus_set.name + "_" + name
        assemblies.append(choice_assembly)
    return assemblies


def unique_preserved_order(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


if __name__ == '__main__':
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    benchmark = load('brain-score')
    source = load_assembly('dicarlo.Majaj2015')
    score = benchmark(source, transformation_kwargs=dict(
        cartesian_product_kwargs=dict(dividing_coord_names_source=['region'])))
    assert score == 1
