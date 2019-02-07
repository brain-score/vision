from brainscore.metrics.neural_predictivity import PlsPredictivity
from result_caching import store

import brainscore
from brainscore.assemblies import walk_coords, array_is_element, merge_data_arrays
from brainscore.benchmarks import Benchmark, AssemblyLoader, _benchmarks, assembly_loaders, mean_over
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.transformations import subset, CartesianProduct


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


class DicarloMajaj2015EarlyLate(Benchmark):
    def __init__(self):
        loader = DicarloMajaj2015EarlyLateLoader()
        assembly_repetitions = loader(average_repetition=False)
        ceiling = InternalConsistency(assembly_repetitions)
        assembly = loader(average_repetition=True)
        metric = PlsPredictivity()
        self._cross_region = CartesianProduct(dividers=['region'])
        self._cross_time = CartesianProduct(dividers=['time_bin_start'])
        super(DicarloMajaj2015EarlyLate, self).__init__(name='dicarlo.Majaj2015.earlylate',
                                                        target_assembly=assembly, metric=metric, ceiling=ceiling)

    def __call__(self, source_assembly):
        # subset the values where the image_ids match up. The stimulus set of this assembly provides
        # all the images (including variations 0 and 3), but the assembly considers only variation 6.
        source_assembly = subset(source_assembly, self._target_assembly, subset_dims=['image_id'])
        score = self._cross_region(self._target_assembly, apply=
        lambda region_assembly: self._cross_time(region_assembly, apply=
        lambda region_time_assembly: self._metric(source_assembly, region_time_assembly)))
        return score


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


class DicarloMajaj2015EarlyLateLoader(AssemblyLoader):
    def __init__(self):
        super(DicarloMajaj2015EarlyLateLoader, self).__init__(name='dicarlo.Majaj2015.earlylate')

    @store()
    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='dicarlo.Majaj2015.temporal')

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
        if average_repetition:
            assembly = mean_over(assembly, presentation=['category_name', 'object_name', 'image_id'])
        return assembly


def inject():
    experimental_loaders = [GallantDavid2004Loader(),
                            ToliasCadena2017Loader()]
    assembly_loaders.update({loader.name: loader for loader in experimental_loaders})
    _benchmarks.update({
        'tolias.Cadena2017': ToliasCadena2017,
        'movshon.FreemanZiemba2013.V1': MovshonFreemanZiemba2013V1,
        'movshon.FreemanZiemba2013.V2': MovshonFreemanZiemba2013V2,
    })
