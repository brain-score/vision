import xarray as xr
from brainscore.metrics.transformations import subset

import brainscore
from brainscore.assemblies import merge_data_arrays, walk_coords, array_is_element
from result_caching import store


class AssemblyLoader(object):
    def __init__(self, name):
        self.name = name

    def __call__(self):
        raise NotImplementedError()


class DicarloMajaj2015Loader(AssemblyLoader):
    def __init__(self):
        super(DicarloMajaj2015Loader, self).__init__(name='dicarlo.Majaj2015')
        self.average_repetition = average_repetition

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name=self.name)
        assembly.load()
        assembly = self._filter_erroneous_neuroids(assembly)
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


class _RegionLoader(AssemblyLoader):
    def __init__(self, basename, region):
        super(_RegionLoader, self).__init__(name=f'{basename}.{region}')
        self.region = region
        self.loader = assembly_loaders[basename]
        self.average_repetition = self.loader.average_repetition

    def __call__(self, average_repetition=True):
        assembly = self.loader(average_repetition=average_repetition)
        assembly.name = self.name
        assembly = assembly.sel(region=self.region)
        if 'neuroid_id' in assembly.dims:  # work around xarray multiindex issues
            assembly = assembly.stack(neuroid=['neuroid_id'])
        assembly['region'] = 'neuroid', [self.region] * len(assembly['neuroid'])
        return assembly


class _VariationLoader(AssemblyLoader):
    def __init__(self, basename, variation_name, variation):
        super(_VariationLoader, self).__init__(name=f'{basename}.{variation_name}var')
        self.variation = variation
        self.loader = assembly_loaders[basename]
        self.average_repetition = self.loader.average_repetition

    def __call__(self, average_repetition=True):
        assembly = self.loader(average_repetition=average_repetition)
        assembly.name = self.name
        variation = [self.variation] if not isinstance(self.variation, list) else self.variation
        variation_selection = xr.DataArray([0] * len(variation), coords={'variation': variation},
                                           dims=['variation']).stack(presentation=['variation'])
        assembly = subset(assembly, variation_selection, repeat=True, dims_must_match=False)
        assert hasattr(assembly, 'variation')
        adapt_stimulus_set(assembly, name_suffix="var" + "".join(str(v) for v in variation))
        return assembly


def adapt_stimulus_set(assembly, name_suffix):
    stimulus_set_name = f"{assembly.stimulus_set.name}-{name_suffix}"
    assembly.attrs['stimulus_set'] = assembly.stimulus_set[
        assembly.stimulus_set['image_id'].isin(assembly['image_id'].values)]
    assembly.stimulus_set.name = stimulus_set_name
    assembly.attrs['stimulus_set_name'] = stimulus_set_name


DicarloMajaj2015LowvarLoader = lambda: _VariationLoader(basename='dicarlo.Majaj2015',
                                                        variation_name='low', variation=[0, 3])
DicarloMajaj2015HighvarLoader = lambda: _VariationLoader(basename='dicarlo.Majaj2015',
                                                         variation_name='high', variation=6)
# separate into mapping and test  # TODO: these need to be packaged separately for access rights
DicarloMajaj2015V4LowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.lowvar', region='V4')
DicarloMajaj2015ITLowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.lowvar', region='IT')
DicarloMajaj2015V4HighvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.highvar', region='V4')
DicarloMajaj2015ITHighvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.highvar', region='IT')


class DicarloMajaj2015TemporalLoader(AssemblyLoader):
    def __init__(self, name='dicarlo.Majaj2015.temporal'):
        super(DicarloMajaj2015TemporalLoader, self).__init__(name=name)
        self._helper = DicarloMajaj2015Loader()

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='dicarlo.Majaj2015.temporal')
        assembly = self._helper._filter_erroneous_neuroids(assembly)
        assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
        if average_repetition:
            assembly = self._helper.average_repetition(assembly)
        return assembly

DicarloMajaj2015TemporalLowvarLoader = lambda: _VariationLoader(basename='dicarlo.Majaj2015.temporal',
                                                        variation_name='low', variation=[0, 3])
DicarloMajaj2015TemporalHighvarLoader = lambda: _VariationLoader(basename='dicarlo.Majaj2015.temporal',
                                                         variation_name='high', variation=6)
# separate into mapping and test  # TODO: these need to be packaged separately for access rights
DicarloMajaj2015TemporalV4LowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.temporal.lowvar', region='V4')
DicarloMajaj2015TemporalITLowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.temporal.lowvar', region='IT')
DicarloMajaj2015TemporalV4HighvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.temporal.highvar', region='V4')
DicarloMajaj2015TemporalITHighvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.temporal.highvar', region='IT')


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


class MovshonFreemanZiemba2013Loader(AssemblyLoader):
    def __init__(self):
        super(MovshonFreemanZiemba2013Loader, self).__init__(name='movshon.FreemanZiemba2013')
        self.average_repetition = average_repetition

    @store()
    def __call__(self, average_repetition=True):
        time_window = (50, 200)
        assembly = brainscore.get_assembly(name='movshon.FreemanZiemba2013')
        assembly.load()
        assembly = assembly.sel(time_bin=[(t, t + 1) for t in range(*time_window)])
        assembly = assembly.mean(dim='time_bin', keep_attrs=True)
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        assembly.stimulus_set.name = assembly.stimulus_set_name
        return assembly


MovshonFreemanZiemba2013V1Loader = lambda: _RegionLoader(basename='movshon.FreemanZiemba2013', region='V1')
MovshonFreemanZiemba2013V2Loader = lambda: _RegionLoader(basename='movshon.FreemanZiemba2013', region='V2')


def average_repetition(assembly):
    def avg_repr(assembly):
        presentation_coords = [coord for coord, dims, values in walk_coords(assembly)
                               if array_is_element(dims, 'presentation') and coord != 'repetition']
        assembly = assembly.multi_groupby(presentation_coords).mean(dim='presentation', skipna=True)
        return assembly

    return apply_keep_attrs(assembly, avg_repr)


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


def apply_keep_attrs(assembly, fnc):  # workaround to keeping attrs
    attrs = assembly.attrs
    assembly = fnc(assembly)
    assembly.attrs = attrs
    return assembly


_assembly_loaders_ctrs = [
    MovshonFreemanZiemba2013Loader, MovshonFreemanZiemba2013V1Loader, MovshonFreemanZiemba2013V2Loader,
    ToliasCadena2017Loader,
    GallantDavid2004Loader,
    DicarloMajaj2015Loader, DicarloMajaj2015LowvarLoader, DicarloMajaj2015HighvarLoader,
    DicarloMajaj2015V4LowvarLoader, DicarloMajaj2015ITLowvarLoader,  # public mapping
    DicarloMajaj2015V4HighvarLoader, DicarloMajaj2015ITHighvarLoader,  # private testing
    DicarloMajaj2015EarlyLateLoader,
    DicarloMajaj2015TemporalLoader,  # private testing temporal loaders
    DicarloMajaj2015TemporalLowvarLoader,
    DicarloMajaj2015TemporalHighvarLoader,
    DicarloMajaj2015TemporalV4LowvarLoader,
    DicarloMajaj2015TemporalITLowvarLoader,
    DicarloMajaj2015TemporalV4HighvarLoader,
    DicarloMajaj2015TemporalITHighvarLoader,
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
