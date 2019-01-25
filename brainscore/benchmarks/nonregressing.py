from collections import OrderedDict

import numpy as np
from numpy.random.mtrand import RandomState
from xarray import DataArray

from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.loaders import DicarloMajaj2015Loader, ToliasCadena2017Loader
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.correlation import CrossCorrelation
from brainscore.metrics.transformations import subset


class DicarloMajaj2015Region(BenchmarkBase):
    def __init__(self, region):
        loader = DicarloMajaj2015Loader()
        assembly_repetitions = loader(average_repetition=False)
        assembly_repetitions = assembly_repetitions.sel(region=region)
        assembly_repetitions = split_assembly(assembly_repetitions,
                                              named_ratios=[('train', .5), ('validation', .3), ('test', .2)])
        # named_ratios=[('train', .9), ('validation', .09), ('test', .01)])
        training_assembly_repetitions, validation_assembly_repetitions, test_assembly_repetitions = assembly_repetitions
        assemblies = [loader.average_repetition(assembly_repetition) for assembly_repetition in assembly_repetitions]
        training_assembly, validation_assembly, test_assembly = assemblies

        similarity_metric = CrossCorrelation()
        name = f'dicarlo.Majaj2015.{region}-nonregressing'
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
