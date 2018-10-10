import logging

import numpy as np
from result_caching import cache, store

import brainscore
from brainscore.assemblies import merge_data_arrays, walk_coords, array_is_element, DataAssembly
from brainscore.metrics import NonparametricWrapper, Score
from brainscore.metrics.anatomy import EdgeRatioMetric
from brainscore.metrics.ceiling import ceilings, InternalConsistency
from brainscore.metrics.neural_predictivity import PlsPredictivity, LinearPredictivity, NeuralPredictivity
from brainscore.metrics.rdm import RDMMetric
from brainscore.utils import fullname


class Benchmark(object):
    def __init__(self, name, target_assembly, metric, ceiling):
        self.name = name
        self._target_assembly = target_assembly
        self.stimulus_set_name = target_assembly.attrs['stimulus_set_name']
        self._metric = metric
        self._ceiling = ceiling
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_assembly, identifier=None):
        if identifier is None and source_assembly.name is None:
            raise ValueError("must provide either identifier or source_assembly.name")
        identifier = identifier or source_assembly.name
        return self._cached_call(source_assembly, identifier=identifier)

    # noinspection PyUnusedLocal
    @store(identifier_ignore=['source_assembly'])
    def _cached_call(self, source_assembly, identifier):
        return self._metric(source_assembly, self._target_assembly)


class BrainScore(object):
    # Brain-Score is a Benchmark too, but due to its compositionality
    # we deem it too different from the Benchmark base class.
    def __init__(self):
        self.name = 'brain-score'
        self._benchmarks = [DicarloMajaj2015V4(), DicarloMajaj2015IT()]
        # hacky but works for now: copy stimulus set from child benchmark.
        # this will not scale to multiple child benchmarks.
        self.stimulus_set_name = self._benchmarks[0].stimulus_set_name
        self._kwargs = None

    def __call__(self, source_assembly, identifier=None, **kwargs):
        scores = [benchmark(source_assembly, **kwargs) for benchmark in self._benchmarks]
        scores = [score.aggregation for score in scores]

        def aggregate_score(score):
            aggregation_dims = ['aggregation', 'region']
            assert all(dim in score.dims for dim in aggregation_dims)
            reduce_dims = [dim for dim in score.dims if dim not in aggregation_dims]
            best_score = score.max(reduce_dims)
            np.testing.assert_array_equal(best_score.dims, aggregation_dims)
            return best_score.sel(aggregation='center')

        scores = [aggregate_score(score) for score in scores]
        brain_score = np.mean(scores).values  # TODO
        # TODO: behavior
        values = [score.expand_dims('benchmark') for score in scores]
        for value, benchmark in zip(values, self._benchmarks):
            value['benchmark'] = [benchmark.name]
        values = merge_data_arrays(values)
        aggregation = DataAssembly([brain_score], coords={'aggregation': ['center']}, dims=['aggregation'])
        score = Score(aggregation=aggregation, values=values)
        return score


class DicarloMajaj2015Region(Benchmark):
    # We here treat V4 and IT as separate benchmarks
    # even though they were collected from the same brains in the same sessions.
    def __init__(self, region):
        loader = DicarloMajaj2015Loader()
        assembly_repetitions = loader(average_repetition=False)
        assembly_repetitions = assembly_repetitions.sel(region=region)
        metric = NeuralPredictivity()
        ceiling = InternalConsistency(assembly_repetitions)
        assembly = loader.average_repetition(assembly_repetitions)
        super(DicarloMajaj2015Region, self).__init__(name=f'dicarlo.Majaj2015.{region}', target_assembly=assembly,
                                                     metric=metric, ceiling=ceiling)


class DicarloMajaj2015V4(DicarloMajaj2015Region):
    def __init__(self):
        super(DicarloMajaj2015V4, self).__init__(region='V4')


class DicarloMajaj2015IT(DicarloMajaj2015Region):
    def __init__(self):
        super(DicarloMajaj2015IT, self).__init__(region='IT')


class ToliasCadena2017(Benchmark):
    def __init__(self):
        loader = ToliasCadena2017Loader()
        assembly_repetitions = loader(average_repetition=False)
        ceiling = InternalConsistency(assembly_repetitions)
        assembly = loader.average_repetition(assembly_repetitions)
        metric = PlsPredictivity()
        super(ToliasCadena2017, self).__init__(name='tolias.Cadena2017', target_assembly=assembly,
                                               metric=metric, ceiling=ceiling)


class DicarloMajaj2015ITEarlyLate(Benchmark):
    def __init__(self):
        loader = DicarloMajaj2015EarlyLateLoader()
        assembly_repetitions = loader(average_repetition=False)
        assembly_repetitions = assembly_repetitions.sel(region='IT')
        ceiling = InternalConsistency(assembly_repetitions)
        assembly = loader.average_repetition(assembly_repetitions)
        metric = PlsPredictivity()
        super(DicarloMajaj2015ITEarlyLate, self).__init__(name='dicarlo.Majaj2015.IT.earlylate',
                                                          target_assembly=assembly, metric=metric, ceiling=ceiling)
        # TODO: target_splits=('region', 'time_bin_start', 'time_bin_end'))


class AssemblyLoader(object):
    def __init__(self, name):
        self.name = name

    def __call__(self):
        raise NotImplementedError()


class DicarloMajaj2015Loader(AssemblyLoader):
    def __init__(self, name='dicarlo.Majaj2015'):
        super(DicarloMajaj2015Loader, self).__init__(name=name)

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name=self.name)
        assembly.load()
        assembly = self._filter_erroneous_neuroids(assembly)
        assembly = assembly.sel(variation=6)  # TODO: remove variation selection once part of name
        assembly = assembly.squeeze("time_bin")
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly

    def _filter_erroneous_neuroids(self, assembly):
        err_neuroids = ['Tito_L_P_8_5', 'Tito_L_P_7_3', 'Tito_L_P_7_5', 'Tito_L_P_5_1', 'Tito_L_P_9_3',
                        'Tito_L_P_6_3', 'Tito_L_P_7_4', 'Tito_L_P_5_0', 'Tito_L_P_5_4', 'Tito_L_P_9_6',
                        'Tito_L_P_0_4', 'Tito_L_P_4_6', 'Tito_L_P_5_6', 'Tito_L_P_7_6', 'Tito_L_P_9_8',
                        'Tito_L_P_4_1', 'Tito_L_P_0_5', 'Tito_L_P_9_9', 'Tito_L_P_3_0', 'Tito_L_P_0_3',
                        'Tito_L_P_6_6', 'Tito_L_P_5_7', 'Tito_L_P_1_1', 'Tito_L_P_3_8', 'Tito_L_P_1_6',
                        'Tito_L_P_3_5', 'Tito_L_P_6_8', 'Tito_L_P_2_8', 'Tito_L_P_9_7', 'Tito_L_P_6_7',
                        'Tito_L_P_1_0', 'Tito_L_P_4_5', 'Tito_L_P_4_9', 'Tito_L_P_7_8', 'Tito_L_P_4_7',
                        'Tito_L_P_4_0', 'Tito_L_P_3_9', 'Tito_L_P_7_7', 'Tito_L_P_4_3', 'Tito_L_P_9_5']
        good_neuroids = [i for i, neuroid_id in enumerate(assembly['neuroid_id'].values)
                         if neuroid_id not in err_neuroids]
        assembly = assembly.isel(neuroid=good_neuroids)
        return assembly

    def average_repetition(self, assembly):
        attrs = assembly.attrs  # workaround to keeping attrs
        assembly = assembly.multi_groupby(['category_name', 'object_name', 'image_id']).mean(dim='presentation')
        assembly.attrs = attrs
        return assembly


class DicarloMajaj2015TemporalLoader(DicarloMajaj2015Loader):
    def __init__(self, name='dicarlo.Majaj2015.temporal'):
        super(DicarloMajaj2015TemporalLoader, self).__init__(name=name)

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='dicarlo.Majaj2015.temporal')
        assembly = self._filter_erroneous_neuroids(assembly)
        assembly = assembly.sel(variation=6)  # TODO: remove variation selection once part of name
        assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly


class DicarloMajaj2015EarlyLateLoader(DicarloMajaj2015TemporalLoader):
    def __init__(self):
        super(DicarloMajaj2015EarlyLateLoader, self).__init__(name='dicarlo.Majaj2015.earlylate')

    @store()
    def __call__(self, average_repetition=True):
        assembly = super().__call__(average_repetition=average_repetition)

        # the principled way here would be to compute the internal consistency per neuron
        # and only use the time bins where the neuron's internal consistency is e.g. >0.2.
        # Note that this would have to be done per neuron
        # as a neuron in V4 will exhibit different time bins from one in IT.
        # We cut off at 200 ms because that's when the next stimulus is shown.
        def sel_time_bin(time_bin_start, time_bin_end):
            selection = assembly.sel(time_bin_start=time_bin_start, time_bin_end=time_bin_end)
            del selection['time_bin']
            selection = selection.expand_dims('time_bin_start').expand_dims('time_bin_end')
            selection['time_bin_start'] = [time_bin_start]
            selection['time_bin_end'] = [time_bin_end]
            selection = selection.stack(time_bin=('time_bin_start', 'time_bin_end'))
            return selection

        early = sel_time_bin(90, 110)
        late = sel_time_bin(190, 210)
        assembly = merge_data_arrays([early, late])
        return assembly


class ToliasCadena2017Loader(AssemblyLoader):
    def __init__(self):
        super(ToliasCadena2017Loader, self).__init__(name='tolias.Cadena2017')

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='tolias.Cadena2017')
        assembly.load()
        assembly = assembly.rename({'neuroid': 'neuroid_id'})
        assembly['region'] = 'neuroid_id', ['V1'] * len(assembly['neuroid_id'])
        assembly = assembly.stack(neuroid=['neuroid_id'])
        assembly = assembly.squeeze("time_bin")
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly

    def average_repetition(self, assembly):
        attrs = assembly.attrs  # workaround to keeping attrs
        presentation_coords = [coord for coord, dims, values in walk_coords(assembly)
                               if array_is_element(dims, 'presentation')]
        presentation_coords = set(presentation_coords) - {'repetition_id', 'id'}
        assembly = assembly.multi_groupby(presentation_coords).mean(dim='presentation', skipna=True)
        assembly = assembly.dropna('presentation')  # discard any images with NaNs (~14%)
        assembly.attrs = attrs
        return assembly


class GallantDavid2004Loader(AssemblyLoader):
    def __init__(self):
        super(GallantDavid2004Loader, self).__init__(name='gallant.David2004')

    def __call__(self):
        assembly = brainscore.get_assembly(name=self.name)
        assembly.load()
        assembly = assembly.rename({'neuroid': 'neuroid_id'})
        assembly = assembly.stack(neuroid=('neuroid_id',))
        assembly = assembly.transpose('presentation', 'neuroid')
        return assembly


metrics = {
    'rdm': lambda *args, **kwargs: NonparametricWrapper(RDMMetric(*args, **kwargs)),
    'linear_predictivity': LinearPredictivity,
    'pls_predictivity': PlsPredictivity,
    'edge_ratio': EdgeRatioMetric
}

assembly_loaders = [DicarloMajaj2015Loader(), DicarloMajaj2015EarlyLateLoader(), GallantDavid2004Loader(),
                    ToliasCadena2017Loader()]
assembly_loaders = {loader.name: loader for loader in assembly_loaders}

_benchmarks = {
    'brain-score': BrainScore,
    'dicarlo.Majaj2015.V4': DicarloMajaj2015V4,
    'dicarlo.Majaj2015.IT': DicarloMajaj2015IT,
    'dicarlo.Majaj2015.IT.earlylate': DicarloMajaj2015ITEarlyLate,
    'tolias.Cadena2017': ToliasCadena2017,
}


@cache()
def load(name):
    if name not in _benchmarks:
        raise ValueError("Unknown benchmark '{}' - must choose from {}".format(name, list(_benchmarks.keys())))
    return _benchmarks[name]()


def load_assembly(name: str):
    """
    Loads the assembly using an AssemblyLoader.
    The AssemblyLoader might further refine the raw assembly provided by brainscore.get_assembly.
    :param name: the name of the assembly loader
    :return: the loaded assembly
    """
    return assembly_loaders[name]()


def build(name, assembly_name, metric_name, ceiling_name=None):
    assembly = load_assembly(assembly_name)
    metric = metrics[metric_name]()
    ceiling = ceilings[ceiling_name]()
    return Benchmark(name=name, target_assembly=assembly, metric=metric, ceiling=ceiling)


if __name__ == '__main__':
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    benchmark = load('brain-score')
    source = load_assembly('dicarlo.Majaj2015')
    score = benchmark(source, transformation_kwargs=dict(
        cartesian_product_kwargs=dict(dividing_coord_names_source=['region'])))
    assert score == 1
