import brainscore
from brainscore.assemblies import merge_data_arrays, walk_coords, array_is_element, AssemblyLoader, average_repetition
from result_caching import store


class DicarloMajaj2015Loader(AssemblyLoader):
    def __init__(self, access):
        super(DicarloMajaj2015Loader, self).__init__(name=f'dicarlo.Majaj2015.{access}')
        self.average_repetition = average_repetition

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name=self.name)
        assembly.load()
        assembly = assembly.squeeze("time_bin")
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly


class _RegionLoader(AssemblyLoader):
    def __init__(self, basename, region, assembly_loader_pool=None):
        super(_RegionLoader, self).__init__(name=f'{basename}.{region}')
        assembly_loader_pool = assembly_loader_pool or assembly_loaders
        self.region = region
        self.loader = assembly_loader_pool[basename]
        self.average_repetition = self.loader.average_repetition

    def __call__(self, average_repetition=True):
        assembly = self.loader(average_repetition=average_repetition)
        assembly.name = self.name
        assembly = assembly.sel(region=self.region)
        if 'neuroid_id' in assembly.dims:  # work around xarray multiindex issues
            assembly = assembly.stack(neuroid=['neuroid_id'])
        assembly['region'] = 'neuroid', [self.region] * len(assembly['neuroid'])
        return assembly


DicarloMajaj2015HighvarLoader = lambda: DicarloMajaj2015Loader(access='private')
DicarloMajaj2015V4HighvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.private', region='V4')
DicarloMajaj2015ITHighvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.private', region='IT')


class DicarloMajaj2015TemporalLoader(AssemblyLoader):
    def __init__(self, access, name='dicarlo.Majaj2015.temporal'):
        super(DicarloMajaj2015TemporalLoader, self).__init__(name=f'{name}.{access}')
        self.access = access
        self.average_repetition = average_repetition

    @store()
    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name=f'dicarlo.Majaj2015.temporal.{self.access}')
        assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly


DicarloMajaj2015TemporalHighvarLoader = lambda: DicarloMajaj2015TemporalLoader(access='private')
DicarloMajaj2015TemporalV4HighvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.temporal.private',
                                                                region='V4')
DicarloMajaj2015TemporalITHighvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.temporal.private',
                                                                region='IT')


class DicarloMajaj2015EarlyLateLoader(DicarloMajaj2015TemporalLoader):
    def __init__(self):
        super(DicarloMajaj2015EarlyLateLoader, self).__init__('private', name='dicarlo.Majaj2015.earlylate')

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


class MovshonFreemanZiemba2013TemporalLoader(AssemblyLoader):
    def __init__(self, access):
        super(MovshonFreemanZiemba2013TemporalLoader, self).__init__(
            name=f'movshon.FreemanZiemba2013.temporal.{access}')
        self.average_repetition = average_repetition
        self.access = access

    @store()
    def __call__(self, average_repetition=True):
        time_bins = [(time_bin_start, time_bin_start + 10) for time_bin_start in range(0, 291, 10)]
        assembly = brainscore.get_assembly(f'movshon.FreemanZiemba2013.{self.access}')
        assembly.load()
        time_assemblies = []
        for time_bin_start, time_bin_end in time_bins:
            time_assembly = assembly.sel(time_bin=[(t, t + 1) for t in range(time_bin_start, time_bin_end)])
            time_assembly = time_assembly.mean(dim='time_bin', keep_attrs=True)
            time_assembly = time_assembly.expand_dims('time_bin_start').expand_dims('time_bin_end')
            time_assembly['time_bin_start'] = [time_bin_start]
            time_assembly['time_bin_end'] = [time_bin_end]
            time_assembly = time_assembly.stack(time_bin=['time_bin_start', 'time_bin_end'])
            time_assemblies.append(time_assembly)
        assembly = merge_data_arrays(time_assemblies)
        assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly


class MovshonFreemanZiemba2013Loader(AssemblyLoader):
    def __init__(self, access):
        super(MovshonFreemanZiemba2013Loader, self).__init__(name=f'movshon.FreemanZiemba2013.{access}')
        self.average_repetition = average_repetition

    @store()
    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name=self.name)
        assembly.load()
        time_window = (50, 200)
        assembly = assembly.sel(time_bin=[(t, t + 1) for t in range(*time_window)])
        assembly = assembly.mean(dim='time_bin', keep_attrs=True)

        assembly = assembly.expand_dims('time_bin_start').expand_dims('time_bin_end')
        assembly['time_bin_start'], assembly['time_bin_end'] = [time_window[0]], [time_window[1]]
        assembly = assembly.stack(time_bin=['time_bin_start', 'time_bin_end'])
        assembly = assembly.squeeze('time_bin')
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly


MovshonFreemanZiemba2013PrivateLoader = lambda: MovshonFreemanZiemba2013Loader('private')
MovshonFreemanZiemba2013TemporalPrivateLoader = lambda: MovshonFreemanZiemba2013TemporalLoader('private')
MovshonFreemanZiemba2013V1PrivateLoader = lambda: _RegionLoader('movshon.FreemanZiemba2013.private', 'V1')
MovshonFreemanZiemba2013V2PrivateLoader = lambda: _RegionLoader('movshon.FreemanZiemba2013.private', 'V2')

MovshonFreemanZiemba2013TemporalV1PrivateLoader = lambda: _RegionLoader('movshon.FreemanZiemba2013.temporal.private',
                                                                        'V1')
MovshonFreemanZiemba2013TemporalV2PrivateLoader = lambda: _RegionLoader('movshon.FreemanZiemba2013.temporal.private',
                                                                        'V2')


class Rajalingham2018Loader(AssemblyLoader):
    def __init__(self):
        super(Rajalingham2018Loader, self).__init__(name='dicarlo.Rajalingham2018')

    def __call__(self):
        assembly = brainscore.get_assembly('dicarlo.Rajalingham2018.private')
        assembly['correct'] = assembly['choice'] == assembly['sample_obj']
        return assembly


class ToliasCadena2017Loader(AssemblyLoader):
    def __init__(self):
        super(ToliasCadena2017Loader, self).__init__(name='tolias.Cadena2017')

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='tolias.Cadena2017')
        assembly = assembly.rename({'neuroid': 'neuroid_id'}).stack(neuroid=['neuroid_id'])
        assembly.load()
        assembly['region'] = 'neuroid', ['V1'] * len(assembly['neuroid'])
        assembly = assembly.squeeze("time_bin")
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly

    def _align_stimuli(self, stimulus_set, image_ids):
        stimulus_set = stimulus_set.loc[stimulus_set['image_id'].isin(image_ids)]
        return stimulus_set

    def average_repetition(self, assembly):
        attrs = assembly.attrs  # workaround to keeping attrs
        presentation_coords = [coord for coord, dims, values in walk_coords(assembly)
                               if array_is_element(dims, 'presentation')]
        presentation_coords = set(presentation_coords) - {'repetition_id', 'id'}
        assembly = assembly.multi_groupby(presentation_coords).mean(dim='presentation', skipna=True)
        assembly, stimulus_set = self.dropna(assembly, stimulus_set=attrs['stimulus_set'])
        attrs['stimulus_set'] = stimulus_set
        assembly.attrs = attrs
        return assembly

    def dropna(self, assembly, stimulus_set):
        assembly = assembly.dropna('presentation')  # discard any images with NaNs (~14%)
        stimulus_set = self._align_stimuli(stimulus_set, assembly.image_id.values)
        return assembly, stimulus_set


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


_assembly_loaders_ctrs = [
    MovshonFreemanZiemba2013PrivateLoader, MovshonFreemanZiemba2013TemporalPrivateLoader,
    MovshonFreemanZiemba2013V1PrivateLoader, MovshonFreemanZiemba2013V2PrivateLoader,
    MovshonFreemanZiemba2013TemporalV1PrivateLoader, MovshonFreemanZiemba2013TemporalV2PrivateLoader,
    ToliasCadena2017Loader,
    GallantDavid2004Loader,
    DicarloMajaj2015HighvarLoader,
    DicarloMajaj2015V4HighvarLoader, DicarloMajaj2015ITHighvarLoader,
    DicarloMajaj2015EarlyLateLoader,
    DicarloMajaj2015TemporalHighvarLoader,
    DicarloMajaj2015TemporalV4HighvarLoader,
    DicarloMajaj2015TemporalITHighvarLoader,
    Rajalingham2018Loader,
]
assembly_loaders = {}
for loader_ctr in _assembly_loaders_ctrs:
    loader = loader_ctr()
    assembly_loaders[loader.name] = loader


def load_assembly(name: str, **kwargs):
    """
    Loads the assembly using an AssemblyLoader.
    The AssemblyLoader might further refine the raw assembly provided by brainscore.get_assembly.
    :param name: the name of the assembly loader
    :return: the loaded assembly
    """
    return assembly_loaders[name](**kwargs)
