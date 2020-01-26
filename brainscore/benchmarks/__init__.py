from abc import ABC

from result_caching import cache, store

from brainscore.metrics import Score
from brainscore.utils import LazyLoad


class Benchmark(ABC):
    def __call__(self, candidate):
        raise NotImplementedError()

    @property
    def identifier(self):
        raise NotImplementedError()

    @property
    def version(self):
        raise NotImplementedError()

    @property
    def ceiling(self):
        raise NotImplementedError()


class BenchmarkBase(Benchmark):
    def __init__(self, identifier, ceiling_func, version, parent=None, paper_link=None):
        self._identifier = identifier
        self._ceiling_func = ceiling_func
        self._version = version
        self.parent = parent
        self.paper_link = paper_link

    @property
    def identifier(self):
        return self._identifier

    @property
    def version(self):
        return self._version

    @property
    def ceiling(self):
        return self._ceiling(identifier=self.identifier)

    @store()
    def _ceiling(self, identifier):
        return self._ceiling_func()


def ceil_score(score, ceiling):
    ceiled_center = score.sel(aggregation='center').values / ceiling.sel(aggregation='center').values
    ceiled_score = type(score)([ceiled_center, score.sel(aggregation='error').values],
                               coords=score.coords, dims=score.dims)
    ceiled_score.attrs[Score.RAW_VALUES_KEY] = score
    ceiled_score.attrs['ceiling'] = ceiling
    return ceiled_score


class BenchmarkPool(dict):
    def __init__(self):
        super(BenchmarkPool, self).__init__()
        # local imports to avoid circular imports
        # neural benchmarks
        from .majaj2015 import DicarloMajaj2015V4PLS, DicarloMajaj2015ITPLS, \
            DicarloMajaj2015V4Mask, DicarloMajaj2015ITMask, \
            DicarloMajaj2015V4RDM, DicarloMajaj2015ITRDM
        self['dicarlo.Majaj2015.V4-pls'] = LazyLoad(DicarloMajaj2015V4PLS)
        self['dicarlo.Majaj2015.IT-pls'] = LazyLoad(DicarloMajaj2015ITPLS)
        self['dicarlo.Majaj2015.V4-mask'] = LazyLoad(DicarloMajaj2015V4Mask)
        self['dicarlo.Majaj2015.IT-mask'] = LazyLoad(DicarloMajaj2015ITMask)
        self['dicarlo.Majaj2015.V4-rdm'] = LazyLoad(DicarloMajaj2015V4RDM)
        self['dicarlo.Majaj2015.IT-rdm'] = LazyLoad(DicarloMajaj2015ITRDM)
        from .freemanziemba2013 import MovshonFreemanZiemba2013V1PLS, MovshonFreemanZiemba2013V2PLS, \
            MovshonFreemanZiemba2013V1RDM, MovshonFreemanZiemba2013V2RDM
        self['movshon.FreemanZiemba2013.V1-pls'] = LazyLoad(MovshonFreemanZiemba2013V1PLS)
        self['movshon.FreemanZiemba2013.V2-pls'] = LazyLoad(MovshonFreemanZiemba2013V2PLS)
        self['movshon.FreemanZiemba2013.V1-rdm'] = LazyLoad(MovshonFreemanZiemba2013V1RDM)
        self['movshon.FreemanZiemba2013.V2-rdm'] = LazyLoad(MovshonFreemanZiemba2013V2RDM)
        from .cadena2017 import ToliasCadena2017PLS, ToliasCadena2017Mask
        self['tolias.Cadena2017-pls'] = LazyLoad(ToliasCadena2017PLS)
        self['tolias.Cadena2017-mask'] = LazyLoad(ToliasCadena2017Mask)
        from .kar2019 import DicarloKar2019OST
        self['dicarlo.Kar2019-ost'] = LazyLoad(DicarloKar2019OST)

        # behavioral benchmarks
        from .rajalingham2018 import DicarloRajalingham2018I2n
        self['dicarlo.Rajalingham2018-i2n'] = LazyLoad(DicarloRajalingham2018I2n)

        # engineering (ML) benchmarks
        from .imagenet import Imagenet2012
        self['fei-fei.Deng2009-top1'] = LazyLoad(Imagenet2012)

        from .search import KlabZhang2018ObjArray
        self['klab.Zhang2018-ObjArray'] = LazyLoad(KlabZhang2018ObjArray)


benchmark_pool = BenchmarkPool()


@cache()
def load(name):
    if name not in benchmark_pool:
        raise ValueError(f"Unknown benchmark '{name}' - must choose from {list(benchmark_pool.keys())}")
    return benchmark_pool[name]
