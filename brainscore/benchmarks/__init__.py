"""
A :class:`~brainscore.benchmarks.Benchmark` runs an experiment on a :class:`~brainscore.model_interface.BrainModel`
and tests the resulting measurements against primate `data <https://github.com/brain-score/brainio_collection>`_.
This comparison is done by a :class:`~brainscore.metrics.Metric` which outputs a score of how well model and data match.
This score is normalized with data ceilings and the benchmark returns this ceiled score.
"""

import itertools
from abc import ABC

from brainscore.metrics import Score
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad
from result_caching import cache, store


class Benchmark(ABC):
    """
    Standard Benchmark interface defining the method interfaces.
    """
    def __call__(self, candidate: BrainModel):
        """
        Evaluate a candidate `BrainModel` and return a :class:`~brainscore.metrics.Score` denoting the brain-likeness of
        the model under this benchmark. Typically this involves reproducing the experiment on the model and then
        comparing model measurements (e.g. neural/behavioral) against recordings from biological subjects (e.g.
        primates) using a :class:`~brainscore.metrics.Metric`. The output of this method is a normalized score between 0
        and 1 where 0 means the model does not match the measurements at all and 1 means the model matches the
        measurements at ceiling level (e.g. if the model obtains a score of 0.8 and the data ceiling is also 0.8, the
        score output by this method should be 1).

        :param candidate: a candidate model implementing the `BrainModel` interface. Benchmarks are agnostic of the
                exact implementation and only interact with models through the methods defined in the interface.
        :return: a :class:`~brainscore.metrics.Score` of how brain-like the candidate model is under this benchmark. The
                score is normalized by this benchmark's ceiling such that 1 means the model matches the data to ceiling
                level.
        """
        raise NotImplementedError()

    @property
    def identifier(self):
        """
        Unique identifier for this benchmark.
        Standard format is `<data identifier>-<metric identifier>`, e.g. `dicarlo.Rajalingham2018-i2n`.

        :return: a unique identifier for this benchmark
        """
        raise NotImplementedError()

    @property
    def version(self):
        """
        Optional, but strongly encouraged.

        :return: a version number that is increased every time the model scores for this benchmark change
                (but not for code changes that do not change scores).
        """
        raise NotImplementedError()

    @property
    def ceiling(self):
        """
        The ceiling of this benchmark. Scores need to be normalized by this value.
        Typically this represents the signal in the data and how well we expect the best possible model to score.

        :return: a Score object, denoting the ceiling of this benchmark.
                Typically has two values indexed by an `aggregation` coordinate:
                `center` for the averaged ceiling value, and `error` for the uncertainty.
        """
        raise NotImplementedError()


class BenchmarkBase(Benchmark):
    """
    Helper class for implementing standard functions of the `Benchmark` interface.
    """
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


# define functions creating the benchmark pools, with local imports to avoid circular imports

def _evaluation_benchmark_pool():
    """"
    Standard benchmarks that are evaluated for the website.
    """
    pool = {}
    # neural benchmarks
    from .majaj2015 import DicarloMajaj2015V4PLS, DicarloMajaj2015ITPLS
    pool['dicarlo.Majaj2015.V4-pls'] = LazyLoad(DicarloMajaj2015V4PLS)
    pool['dicarlo.Majaj2015.IT-pls'] = LazyLoad(DicarloMajaj2015ITPLS)
    from .freemanziemba2013 import MovshonFreemanZiemba2013V1PLS, MovshonFreemanZiemba2013V2PLS
    pool['movshon.FreemanZiemba2013.V1-pls'] = LazyLoad(MovshonFreemanZiemba2013V1PLS)
    pool['movshon.FreemanZiemba2013.V2-pls'] = LazyLoad(MovshonFreemanZiemba2013V2PLS)
    from .kar2019 import DicarloKar2019OST
    pool['dicarlo.Kar2019-ost'] = LazyLoad(DicarloKar2019OST)

    # behavioral benchmarks
    from .rajalingham2018 import DicarloRajalingham2018I2n
    pool['dicarlo.Rajalingham2018-i2n'] = LazyLoad(DicarloRajalingham2018I2n)

    return pool


def _engineering_benchmark_pool():
    """
    Additional engineering (ML) benchmarks. These benchmarks are public, but are also be evaluated for the website.
    """
    pool = {}

    from .imagenet import Imagenet2012
    pool['fei-fei.Deng2009-top1'] = LazyLoad(Imagenet2012)

    return pool


def _experimental_benchmark_pool():
    """
    Benchmarks that can be used, but are not evaluated for the website.
    """
    pool = {}
    # neural benchmarks
    from .majaj2015 import DicarloMajaj2015V4Mask, DicarloMajaj2015ITMask, \
        DicarloMajaj2015V4RDM, DicarloMajaj2015ITRDM
    pool['dicarlo.Majaj2015.V4-mask'] = LazyLoad(DicarloMajaj2015V4Mask)
    pool['dicarlo.Majaj2015.IT-mask'] = LazyLoad(DicarloMajaj2015ITMask)
    pool['dicarlo.Majaj2015.V4-rdm'] = LazyLoad(DicarloMajaj2015V4RDM)
    pool['dicarlo.Majaj2015.IT-rdm'] = LazyLoad(DicarloMajaj2015ITRDM)
    from .freemanziemba2013 import MovshonFreemanZiemba2013V1RDM, MovshonFreemanZiemba2013V2RDM, \
        MovshonFreemanZiemba2013V1Single
    pool['movshon.FreemanZiemba2013.V1-rdm'] = LazyLoad(MovshonFreemanZiemba2013V1RDM)
    pool['movshon.FreemanZiemba2013.V2-rdm'] = LazyLoad(MovshonFreemanZiemba2013V2RDM)
    pool['movshon.FreemanZiemba2013.V1-single'] = LazyLoad(MovshonFreemanZiemba2013V1Single)
    from .cadena2017 import ToliasCadena2017PLS, ToliasCadena2017Mask
    pool['tolias.Cadena2017-pls'] = LazyLoad(ToliasCadena2017PLS)
    pool['tolias.Cadena2017-mask'] = LazyLoad(ToliasCadena2017Mask)

    from .search import KlabZhang2018ObjSearchObjArr, KlabZhang2018VisualSearchWaldo, KlabZhang2018VisualSearchNaturaldesign
    pool['klab.Zhang2018.ObjSearch-objarr'] = LazyLoad(KlabZhang2018ObjSearchObjArr)
    pool['klab.Zhang2018.VisualSearch-waldo'] = LazyLoad(KlabZhang2018VisualSearchWaldo)
    pool['klab.Zhang2018.VisualSearch-naturaldesign'] = LazyLoad(KlabZhang2018VisualSearchNaturaldesign)

    return pool


def _public_benchmark_pool():
    """
    Benchmarks that are publicly usable, but are not used for the website.
    """
    pool = {}
    # neural benchmarks
    from .public_benchmarks import FreemanZiembaV1PublicBenchmark, FreemanZiembaV2PublicBenchmark, \
        MajajV4PublicBenchmark, MajajITPublicBenchmark
    pool['movshon.FreemanZiemba2013public.V1-pls'] = LazyLoad(FreemanZiembaV1PublicBenchmark)
    pool['movshon.FreemanZiemba2013public.V2-pls'] = LazyLoad(FreemanZiembaV2PublicBenchmark)
    pool['dicarlo.Majaj2015public.V4-pls'] = LazyLoad(MajajV4PublicBenchmark)
    pool['dicarlo.Majaj2015public.IT-pls'] = LazyLoad(MajajITPublicBenchmark)

    # behavioral benchmarks
    from .public_benchmarks import RajalinghamMatchtosamplePublicBenchmark
    pool['dicarlo.Rajalingham2018public-i2n'] = LazyLoad(RajalinghamMatchtosamplePublicBenchmark)

    return pool


evaluation_benchmark_pool = _evaluation_benchmark_pool()
engineering_benchmark_pool = _engineering_benchmark_pool()
experimental_benchmark_pool = _experimental_benchmark_pool()
public_benchmark_pool = _public_benchmark_pool()


# make sure no identifiers overlap
def check_all_disjoint(*pools):
    union = list(itertools.chain([pool.keys() for pool in pools]))
    duplicates = set([identifier for identifier in union if union.count(identifier) > 1])
    if duplicates:
        raise ValueError(f"Duplicate identifiers in pools: {duplicates}")


check_all_disjoint(evaluation_benchmark_pool, engineering_benchmark_pool,
                   experimental_benchmark_pool, public_benchmark_pool)

# engineering benchmarks are part of both the public as well as the private evaluation pools
public_benchmark_pool = {**public_benchmark_pool, **engineering_benchmark_pool}
evaluation_benchmark_pool = {**evaluation_benchmark_pool, **engineering_benchmark_pool}
# provide unifying pool
benchmark_pool = {**public_benchmark_pool, **engineering_benchmark_pool,
                  **experimental_benchmark_pool, **evaluation_benchmark_pool}


@cache()
def load(name):
    if name not in benchmark_pool:
        raise ValueError(f"Unknown benchmark '{name}' - must choose from {list(benchmark_pool.keys())}")
    return benchmark_pool[name]
