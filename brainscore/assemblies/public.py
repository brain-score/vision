from brainscore.assemblies.private import _VariationLoader, _RegionLoader, _SeparateMovshonPrivatePublic

# Majaj2015 time-averaged
DicarloMajaj2015LowvarLoader = lambda: _VariationLoader(basename='dicarlo.Majaj2015',
                                                        variation_name='low', variation=[0, 3])
DicarloMajaj2015V4LowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.lowvar', region='V4',
                                                       assembly_loader_pool=assembly_loaders)
DicarloMajaj2015ITLowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.lowvar', region='IT',
                                                       assembly_loader_pool=assembly_loaders)

# Majaj2015 temporal
DicarloMajaj2015TemporalLowvarLoader = lambda: _VariationLoader(basename='dicarlo.Majaj2015.temporal',
                                                                variation_name='low', variation=[0, 3])
DicarloMajaj2015TemporalV4LowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.temporal.lowvar',
                                                               region='V4', assembly_loader_pool=assembly_loaders)
DicarloMajaj2015TemporalITLowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.temporal.lowvar',
                                                               region='IT', assembly_loader_pool=assembly_loaders)

# FreemanZiemba2013
MovshonFreemanZiemba2013V1PublicLoader = lambda: _SeparateMovshonPrivatePublic('movshon.FreemanZiemba2013', 'V1',
                                                                               'public')
MovshonFreemanZiemba2013V2PublicLoader = lambda: _SeparateMovshonPrivatePublic('movshon.FreemanZiemba2013', 'V2',
                                                                               'public')

# FreemanZiemba2013 temporal
MovshonFreemanZiemba2013TemporalV1PublicLoader = lambda: _SeparateMovshonPrivatePublic(
    'movshon.FreemanZiemba2013.temporal', 'V1', 'public')
MovshonFreemanZiemba2013TemporalV2PublicLoader = lambda: _SeparateMovshonPrivatePublic(
    'movshon.FreemanZiemba2013.temporal', 'V2', 'public')

_assembly_loaders_ctrs = [
    MovshonFreemanZiemba2013V1PublicLoader, MovshonFreemanZiemba2013V2PublicLoader,
    MovshonFreemanZiemba2013TemporalV1PublicLoader, MovshonFreemanZiemba2013TemporalV2PublicLoader,
    DicarloMajaj2015LowvarLoader, DicarloMajaj2015V4LowvarLoader, DicarloMajaj2015ITLowvarLoader,
    DicarloMajaj2015TemporalLowvarLoader, DicarloMajaj2015TemporalV4LowvarLoader,
    DicarloMajaj2015TemporalITLowvarLoader,
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
