import logging
from abc import ABC

from result_caching import cache, store

from brainscore.benchmarks.loaders import load_assembly, DicarloMajaj2015Loader, ToliasCadena2017Loader
from brainscore.metrics import Score
from brainscore.utils import fullname, LazyLoad


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
    def __init__(self, name, assembly, similarity_metric, ceiling_func):
        self._name = name
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        self._ceiling_func = ceiling_func
        self._logger = logging.getLogger(fullname(self))

    @property
    def name(self):
        return self._name

    @property
    def assembly(self):
        return self._assembly

    def __call__(self, source_assembly):
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        return ceil_score(raw_score, self.ceiling)

    @property
    @store()
    def ceiling(self):
        return self._ceiling_func()


def ceil_score(score, ceiling):
    ceiled_score = score / ceiling
    ceiled_score.attrs[Score.RAW_VALUES_KEY] = score
    return ceiled_score


class BenchmarkPool(dict):
    def __init__(self):
        super(BenchmarkPool, self).__init__()
        # separate into class to avoid circular imports
        from .regressing import DicarloMajaj2015V4, DicarloMajaj2015IT
        self['dicarlo.Majaj2015.V4-regressing'] = LazyLoad(lambda: DicarloMajaj2015V4())
        self['dicarlo.Majaj2015.IT-regressing'] = LazyLoad(lambda: DicarloMajaj2015IT())


benchmark_pool = BenchmarkPool()


@cache()
def load(name):
    if name not in benchmark_pool:
        raise ValueError("Unknown benchmark '{}' - must choose from {}".format(name, list(benchmark_pool.keys())))
    return benchmark_pool[name]
