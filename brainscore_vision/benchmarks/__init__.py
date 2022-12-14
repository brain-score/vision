"""
A :class:`~brainscore.benchmarks.Benchmark` runs an experiment on a :class:`~brainscore.model_interface.BrainModel`
and tests the resulting measurements against primate `data <https://github.com/brain-score/brainio>`_.
This comparison is done by a :class:`~brainscore.metrics.Metric` which outputs a score of how well model and data match.
This score is normalized with data ceilings and the benchmark returns this ceiled score.
"""
import itertools
from abc import ABC

from result_caching import cache, store

from brainscore_vision.metrics import Score
from brainscore_vision.model_interface import BrainModel


class Benchmark(ABC):
    """
    Standard Benchmark interface defining the method interfaces.
    """

    def __call__(self, candidate: BrainModel) -> Score:
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
    def bibtex(self) -> str:
        """
        bibtex string to build the reference.
        Should include an `url` to build a proper link.
        """
        raise NotImplementedError()

    @property
    def identifier(self) -> str:
        """
        Unique identifier for this benchmark.
        Standard format is `<data identifier>-<metric identifier>`, e.g. `dicarlo.Rajalingham2018-i2n`.

        :return: a unique identifier for this benchmark
        """
        raise NotImplementedError()

    @property
    def version(self) -> str:
        """
        Optional, but strongly encouraged.

        :return: a version number that is increased every time the model scores for this benchmark change
                (but not for code changes that do not change scores).
        """
        raise NotImplementedError()

    @property
    def ceiling(self) -> Score:
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

    def __init__(self, identifier, ceiling_func, version, parent=None, bibtex=None):
        self._identifier = identifier
        self._ceiling_func = ceiling_func
        self._version = version
        self.parent = parent
        self._bibtex = bibtex

    @property
    def bibtex(self):
        return self._bibtex

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

    # behavioral benchmarks

    return pool


def _engineering_benchmark_pool():
    """
    Additional engineering (ML) benchmarks. These benchmarks are public, but are also be evaluated for the website.
    """
    pool = {}



    return pool


def _experimental_benchmark_pool():
    """
    Benchmarks that can be used, but are not evaluated for the website.
    """
    pool = {}
    return pool


def _public_benchmark_pool():
    """
    Benchmarks that are publicly usable, but are not used for the website.
    """
    pool = {}
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
evaluation_benchmark_pool = {**evaluation_benchmark_pool}
# provide unifying pool
benchmark_pool = {**public_benchmark_pool, **engineering_benchmark_pool,
                  **experimental_benchmark_pool, **evaluation_benchmark_pool}


@cache()
def load(name):
    if name not in benchmark_pool:
        raise ValueError(f"Unknown benchmark '{name}' - must choose from {list(benchmark_pool.keys())}")
    return benchmark_pool[name]
