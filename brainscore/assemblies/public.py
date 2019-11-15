from brainscore.assemblies.private import _RegionLoader, \
    DicarloMajaj2015Loader, _DicarloMajaj2015TemporalLoader, \
    _MovshonFreemanZiemba2013TemporalLoader, _MovshonFreemanZiemba2013Loader

# Majaj2015 time-averaged
DicarloMajaj2015LowvarLoader = lambda: DicarloMajaj2015Loader(access='public')

DicarloMajaj2015V4LowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.public', region='V4',
                                                       assembly_loader_pool=assembly_loaders)
DicarloMajaj2015ITLowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.public', region='IT',
                                                       assembly_loader_pool=assembly_loaders)


# Majaj2015 temporal
class DicarloMajaj2015TemporalLowvarLoader(_DicarloMajaj2015TemporalLoader):
    def __init__(self):
        super(DicarloMajaj2015TemporalLowvarLoader, self).__init__(access='public')


DicarloMajaj2015TemporalV4LowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.temporal.public',
                                                               region='V4', assembly_loader_pool=assembly_loaders)
DicarloMajaj2015TemporalITLowvarLoader = lambda: _RegionLoader(basename='dicarlo.Majaj2015.temporal.public',
                                                               region='IT', assembly_loader_pool=assembly_loaders)


# FreemanZiemba2013
class MovshonFreemanZiemba2013PublicLoader(_MovshonFreemanZiemba2013Loader):
    def __init__(self):
        super(MovshonFreemanZiemba2013PublicLoader, self).__init__(access='public')


MovshonFreemanZiemba2013V1PublicLoader = lambda: _RegionLoader(
    basename='movshon.FreemanZiemba2013.public', region='V1', assembly_loader_pool=assembly_loaders)
MovshonFreemanZiemba2013V2PublicLoader = lambda: _RegionLoader(
    basename='movshon.FreemanZiemba2013.public', region='V2', assembly_loader_pool=assembly_loaders)


# FreemanZiemba2013 temporal
class MovshonFreemanZiemba2013TemporalPublicLoader(_MovshonFreemanZiemba2013TemporalLoader):
    def __init__(self):
        super(MovshonFreemanZiemba2013TemporalPublicLoader, self).__init__(access='public')


MovshonFreemanZiemba2013TemporalV1PublicLoader = lambda: _RegionLoader(
    basename='movshon.FreemanZiemba2013.temporal.public', region='V1', assembly_loader_pool=assembly_loaders)
MovshonFreemanZiemba2013TemporalV2PublicLoader = lambda: _RegionLoader(
    basename='movshon.FreemanZiemba2013.temporal.public', region='V2', assembly_loader_pool=assembly_loaders)

_assembly_loaders_ctrs = [
    MovshonFreemanZiemba2013PublicLoader, MovshonFreemanZiemba2013TemporalPublicLoader,
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
