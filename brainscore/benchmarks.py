import logging

from result_caching import cache, store

import brainscore
from brainscore.metrics.anatomy import EdgeRatioMetric
from brainscore.metrics.ceiling import ceilings, InternalConsistency
from brainscore.metrics.neural_predictivity import PlsPredictivity, LinearPredictivity
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.transformations import subset
from brainscore.utils import fullname
from brainscore.contrib import benchmarks as contrib_benchmarks


class Benchmark(object):
    def __init__(self, name, target_assembly, metric, ceiling):
        self.name = name
        self._target_assembly = target_assembly
        self.stimulus_set_name = target_assembly.attrs['stimulus_set_name']
        self._metric = metric
        self._ceiling = ceiling
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_assembly):
        return self._metric(source_assembly, self._target_assembly)

    @property
    @store()
    def ceiling(self):
        return self._ceiling()


class _DicarloMajaj2015Region(Benchmark):
    # We here treat V4 and IT as separate benchmarks
    # even though they were collected from the same brains in the same sessions.
    def __init__(self, region):
        loader = DicarloMajaj2015Loader()
        assembly_repetitions = loader(average_repetition=False)
        assembly_repetitions = assembly_repetitions.sel(region=region)
        metric = PlsPredictivity()
        ceiling = InternalConsistency(assembly_repetitions)
        assembly = loader.average_repetition(assembly_repetitions)
        super(_DicarloMajaj2015Region, self).__init__(name=f'dicarlo.Majaj2015.{region}', target_assembly=assembly,
                                                      metric=metric, ceiling=ceiling)

    def __call__(self, source_assembly):
        # subset the values where the image_ids match up. The stimulus set of this assembly provides
        # all the images (including variations 0 and 3), but the assembly considers only variation 6.
        source_assembly = subset(source_assembly, self._target_assembly, subset_dims=['image_id'])
        return super(_DicarloMajaj2015Region, self).__call__(source_assembly=source_assembly)


class DicarloMajaj2015V4(_DicarloMajaj2015Region):
    def __init__(self):
        super(DicarloMajaj2015V4, self).__init__(region='V4')


class DicarloMajaj2015IT(_DicarloMajaj2015Region):
    def __init__(self):
        super(DicarloMajaj2015IT, self).__init__(region='IT')


class AssemblyLoader(object):
    def __init__(self, name):
        self.name = name

    def __call__(self):
        raise NotImplementedError()


class DicarloMajaj2015Loader(AssemblyLoader):
    def __init__(self, name='dicarlo.Majaj2015'):
        super(DicarloMajaj2015Loader, self).__init__(name=name)
        self.average_repetition = lambda assembly: mean_over(assembly,
                                                             presentation=['category_name', 'object_name', 'image_id'])

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name=self.name)
        assembly.load()
        assembly = self._filter_erroneous_neuroids(assembly)
        assembly = assembly.sel(variation=6)
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


def mean_over(assembly, **dim_coords):
    attrs = assembly.attrs  # workaround to keeping attrs
    for dim, coords in dim_coords.items():
        assembly = assembly.multi_groupby(coords).mean(dim=dim)
    assembly.attrs = attrs
    return assembly


metrics = {
    'rdm': RDMCrossValidated,
    'linear_predictivity': LinearPredictivity,
    'pls_predictivity': PlsPredictivity,
    'edge_ratio': EdgeRatioMetric
}

assembly_loaders = [DicarloMajaj2015Loader()]
assembly_loaders = {loader.name: loader for loader in assembly_loaders}

_benchmarks = {
    'dicarlo.Majaj2015.V4': DicarloMajaj2015V4,
    'dicarlo.Majaj2015.IT': DicarloMajaj2015IT,
}

contrib_benchmarks.inject()


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
