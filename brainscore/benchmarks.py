import logging

import brainscore
from brainscore.assemblies import merge_data_arrays, walk_coords, array_is_element
from brainscore.metrics.anatomy import EdgeRatioMetric
from brainscore.metrics.ceiling import ceilings, InternalConsistency
from brainscore.metrics.neural_predictivity import PlsPredictivity, LinearPredictivity
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.transformations import subset
from brainscore.utils import fullname
from result_caching import cache, store


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


class ToliasCadena2017(Benchmark):
    def __init__(self):
        loader = ToliasCadena2017Loader()
        assembly_repetitions = loader(average_repetition=False)
        ceiling = InternalConsistency(assembly_repetitions, split_coord='repetition_id')
        assembly = loader.average_repetition(assembly_repetitions)
        metric = PlsPredictivity()
        super(ToliasCadena2017, self).__init__(name='tolias.Cadena2017', target_assembly=assembly,
                                               metric=metric, ceiling=ceiling)

    def __call__(self, source_assembly):
        # subset the values where the image_ids match up. The stimulus set of this assembly provides
        # more images than actually have good recordings attached.
        source_assembly = subset(source_assembly, self._target_assembly, subset_dims=['image_id'])
        return super(ToliasCadena2017, self).__call__(source_assembly=source_assembly)


class _MovshonFreemanZiemba2013Region(Benchmark):
    def __init__(self, region):
        loader = MovshonFreemanZiemba2013Loader()
        assembly_repetitions = loader(average_repetition=False)
        assembly_repetitions = assembly_repetitions.sel(region=region)
        ceiling = InternalConsistency(assembly_repetitions, split_coord='repetition_id')
        assembly = loader().sel(region=region).stack(neuroid=['neuroid_id'])
        metric = PlsPredictivity()
        super(_MovshonFreemanZiemba2013Region, self).__init__(
            name=f'movshon.FreemanZiemba2013.{region}', target_assembly=assembly, metric=metric, ceiling=ceiling)


class MovshonFreemanZiemba2013V1(_MovshonFreemanZiemba2013Region):
    def __init__(self):
        super(MovshonFreemanZiemba2013V1, self).__init__(region='V1')


class MovshonFreemanZiemba2013V2(_MovshonFreemanZiemba2013Region):
    def __init__(self):
        super(MovshonFreemanZiemba2013V2, self).__init__(region='V2')


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
        assembly = assembly.sel(variation=6)
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


class MovshonFreemanZiemba2013Loader(AssemblyLoader):
    def __init__(self):
        super(MovshonFreemanZiemba2013Loader, self).__init__(name='movshon.FreemanZiemba2013')

    @store()
    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='movshon.FreemanZiemba2013')
        assembly.load()
        # TODO: determine response onset or just take e.g. 40-100?
        assembly = assembly.sel(time_bin=[(t, t + 1) for t in range(40, 100)])
        assembly = assembly.mean(dim='time_bin', keep_attrs=True)
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly

    def average_repetition(self, assembly):
        attrs = assembly.attrs  # workaround to keeping attrs
        presentation_coords = [coord for coord, dims, values in walk_coords(assembly)
                               if array_is_element(dims, 'presentation')]
        presentation_coords = set(presentation_coords) - {'repetition'}
        assembly = assembly.multi_groupby(presentation_coords).mean(dim='presentation', skipna=True)
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
    'rdm': RDMCrossValidated,
    'linear_predictivity': LinearPredictivity,
    'pls_predictivity': PlsPredictivity,
    'edge_ratio': EdgeRatioMetric
}

assembly_loaders = [DicarloMajaj2015Loader(), DicarloMajaj2015EarlyLateLoader(), GallantDavid2004Loader(),
                    ToliasCadena2017Loader()]
assembly_loaders = {loader.name: loader for loader in assembly_loaders}

_benchmarks = {
    'dicarlo.Majaj2015.V4': DicarloMajaj2015V4,
    'dicarlo.Majaj2015.IT': DicarloMajaj2015IT,
    'dicarlo.Majaj2015.IT.earlylate': DicarloMajaj2015ITEarlyLate,
    'tolias.Cadena2017': ToliasCadena2017,
    'movshon.FreemanZiemba2013.V1': MovshonFreemanZiemba2013V1,
    'movshon.FreemanZiemba2013.V2': MovshonFreemanZiemba2013V2,
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
