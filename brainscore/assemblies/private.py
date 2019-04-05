import numpy as np
import xarray as xr
from numpy.random.mtrand import RandomState
from result_caching import store
from sklearn.model_selection import StratifiedShuffleSplit
from xarray import DataArray

import brainscore
from brainscore.assemblies import merge_data_arrays, walk_coords, array_is_element, AssemblyLoader, average_repetition
from brainscore.metrics.transformations import subset
from brainscore.utils import LazyLoad


# TODO: assemblies in here need separate S3 access rights


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


class _VariationLoader(AssemblyLoader):
    def __init__(self, basename, variation_name, variation, assembly_loader_pool=None):
        super(_VariationLoader, self).__init__(name=f'{basename}.{variation_name}var')
        assembly_loader_pool = assembly_loader_pool or assembly_loaders
        self.variation = variation
        self.loader = assembly_loader_pool[basename]
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


DicarloMajaj2015HighvarLoader = lambda: _VariationLoader(basename='dicarlo.Majaj2015',
                                                         variation_name='high', variation=6)
DicarloMajaj2015V4HighvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.highvar', region='V4')
DicarloMajaj2015ITHighvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.highvar', region='IT')


class DicarloMajaj2015TemporalLoader(AssemblyLoader):
    def __init__(self, name='dicarlo.Majaj2015.temporal'):
        super(DicarloMajaj2015TemporalLoader, self).__init__(name=name)
        self._helper = DicarloMajaj2015Loader()
        self.average_repetition = self._helper.average_repetition

    @store()
    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='dicarlo.Majaj2015.temporal')
        assembly = self._helper._filter_erroneous_neuroids(assembly)
        assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
        if average_repetition:
            assembly = self.average_repetition(assembly)
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


def _separate_movshon_private_public(region):
    baseloader = _RegionLoader(basename='movshon.FreemanZiemba2013', region=region)
    base_assembly = baseloader()
    _, unique_indices = np.unique(base_assembly['image_id'].values, return_index=True)
    unique_indices = np.sort(unique_indices)  # preserve order
    image_ids = base_assembly['image_id'].values[unique_indices]
    stratification_values = base_assembly['texture_type'].values[unique_indices]
    rng = RandomState(seed=12)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=.3, test_size=None, random_state=rng)
    split = next(splitter.split(np.zeros(len(image_ids)), stratification_values))
    result = {}
    for assembly_type, image_indices in zip(['public', 'private'], split):
        current_image_ids = image_ids[image_indices]
        subset_indexer = DataArray(np.zeros(len(current_image_ids)), coords={'image_id': current_image_ids},
                                   dims=['image_id']).stack(presentation=['image_id'])
        assembly = subset(base_assembly, subset_indexer, dims_must_match=False)
        adapt_stimulus_set(assembly, assembly_type)
        result[assembly_type] = assembly
    return result


movshon_private_public = {
    'V1': LazyLoad(lambda: _separate_movshon_private_public('V1')),
    'V2': LazyLoad(lambda: _separate_movshon_private_public('V2')),
}


class _SeparateMovshonPrivatePublic(AssemblyLoader):
    def __init__(self, region, private_or_public):
        basename = f"movshon.FreemanZiemba2013.{region}"
        self.region = region
        self.access = private_or_public
        super(_SeparateMovshonPrivatePublic, self).__init__(name=basename)

    def __call__(self):
        return movshon_private_public[self.region][self.access]


MovshonFreemanZiemba2013V1PrivateLoader = lambda: _SeparateMovshonPrivatePublic('V1', 'private')
MovshonFreemanZiemba2013V2PrivateLoader = lambda: _SeparateMovshonPrivatePublic('V2', 'private')


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


_assembly_loaders_ctrs = [
    MovshonFreemanZiemba2013Loader, MovshonFreemanZiemba2013V1PrivateLoader, MovshonFreemanZiemba2013V2PrivateLoader,
    ToliasCadena2017Loader,
    GallantDavid2004Loader,
    DicarloMajaj2015Loader, DicarloMajaj2015HighvarLoader,
    DicarloMajaj2015V4HighvarLoader, DicarloMajaj2015ITHighvarLoader,
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
