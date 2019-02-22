from abc import ABC

from result_caching import cache, store

from brainscore.metrics import Score
from brainscore.utils import LazyLoad


class Benchmark(ABC):
    def __call__(self, candidate):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def ceiling(self):
        raise NotImplementedError()


class BenchmarkBase(Benchmark):
    def __init__(self, name, ceiling_func):
        self._name = name
        self._ceiling_func = ceiling_func

    @property
    def name(self):
        return self._name

    @property
    def ceiling(self):
        return self._ceiling(identifier=self.name)

    @store()
    def _ceiling(self, identifier):
        return self._ceiling_func()


def ceil_score(score, ceiling):
    ceiled_score = score / ceiling
    ceiled_score.attrs[Score.RAW_VALUES_KEY] = score
    ceiled_score.attrs['ceiling'] = ceiling
    return ceiled_score


class BenchmarkPool(dict):
    def __init__(self):
        super(BenchmarkPool, self).__init__()
        # avoid circular imports
        from .regressing import \
            DicarloMajaj2015V4, DicarloMajaj2015IT, \
            MovshonFreemanZiemba2013V1, MovshonFreemanZiemba2013V2
        self['dicarlo.Majaj2015.V4-regressing'] = LazyLoad(lambda: DicarloMajaj2015V4())
        self['dicarlo.Majaj2015.IT-regressing'] = LazyLoad(lambda: DicarloMajaj2015IT())
        self['movshon.FreemanZiemba2013.V1-regressing'] = LazyLoad(lambda: MovshonFreemanZiemba2013V1())
        self['movshon.FreemanZiemba2013.V2-regressing'] = LazyLoad(lambda: MovshonFreemanZiemba2013V2())


benchmark_pool = BenchmarkPool()


@cache()
def load(name):
    if name not in benchmark_pool:
        raise ValueError("Unknown benchmark '{}' - must choose from {}".format(name, list(benchmark_pool.keys())))
    return benchmark_pool[name]
