from result_caching import store

import brainscore
from brainscore.assemblies import walk_coords, array_is_element
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.neural_predictivity import PlsPredictivity
from brainscore.metrics.transformations import subset
from brainscore.benchmarks import Benchmark, AssemblyLoader, _benchmarks, assembly_loaders


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
        assembly_repetitions = assembly_repetitions.sel(region=region).stack(neuroid=['neuroid_id'])
        ceiling = InternalConsistency(assembly_repetitions)
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


def inject():
    experimental_loaders = [GallantDavid2004Loader(),
                            ToliasCadena2017Loader()]
    assembly_loaders.update({loader.name: loader for loader in experimental_loaders})
    _benchmarks.update({
        'tolias.Cadena2017': ToliasCadena2017,
        'movshon.FreemanZiemba2013.V1': MovshonFreemanZiemba2013V1,
        'movshon.FreemanZiemba2013.V2': MovshonFreemanZiemba2013V2,
    })
