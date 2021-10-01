"""
A :class:`~brainscore.benchmarks.Benchmark` runs an experiment on a :class:`~brainscore.model_interface.BrainModel`
and tests the resulting measurements against primate `data <https://github.com/brain-score/brainio>`_.
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
    def bibtex(self):
        """
        bibtex string to build the reference.
        Should include an `url` to build a proper link.
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
    from .majajhong2015 import DicarloMajajHong2015V4PLS, DicarloMajajHong2015ITPLS
    pool['dicarlo.MajajHong2015.V4-pls'] = LazyLoad(DicarloMajajHong2015V4PLS)
    pool['dicarlo.MajajHong2015.IT-pls'] = LazyLoad(DicarloMajajHong2015ITPLS)
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

    from .imagenet_c import Imagenet_C_Noise, Imagenet_C_Blur, Imagenet_C_Weather, Imagenet_C_Digital
    pool['dietterich.Hendrycks2019-noise-top1'] = LazyLoad(Imagenet_C_Noise)
    pool['dietterich.Hendrycks2019-blur-top1'] = LazyLoad(Imagenet_C_Blur)
    pool['dietterich.Hendrycks2019-weather-top1'] = LazyLoad(Imagenet_C_Weather)
    pool['dietterich.Hendrycks2019-digital-top1'] = LazyLoad(Imagenet_C_Digital)

    return pool


def _experimental_benchmark_pool():
    """
    Benchmarks that can be used, but are not evaluated for the website.
    """
    pool = {}
    # neural benchmarks
    from .majajhong2015 import DicarloMajajHong2015V4Mask, DicarloMajajHong2015ITMask, \
        DicarloMajajHong2015V4RDM, DicarloMajajHong2015ITRDM
    pool['dicarlo.MajajHong2015.V4-mask'] = LazyLoad(DicarloMajajHong2015V4Mask)
    pool['dicarlo.MajajHong2015.IT-mask'] = LazyLoad(DicarloMajajHong2015ITMask)
    pool['dicarlo.MajajHong2015.V4-rdm'] = LazyLoad(DicarloMajajHong2015V4RDM)
    pool['dicarlo.MajajHong2015.IT-rdm'] = LazyLoad(DicarloMajajHong2015ITRDM)
    from .freemanziemba2013 import MovshonFreemanZiemba2013V1RDM, MovshonFreemanZiemba2013V2RDM, \
        MovshonFreemanZiemba2013V1Single
    pool['movshon.FreemanZiemba2013.V1-rdm'] = LazyLoad(MovshonFreemanZiemba2013V1RDM)
    pool['movshon.FreemanZiemba2013.V2-rdm'] = LazyLoad(MovshonFreemanZiemba2013V2RDM)
    pool['movshon.FreemanZiemba2013.V1-single'] = LazyLoad(MovshonFreemanZiemba2013V1Single)
    from .cadena2017 import ToliasCadena2017PLS, ToliasCadena2017Mask
    pool['tolias.Cadena2017-pls'] = LazyLoad(ToliasCadena2017PLS)
    pool['tolias.Cadena2017-mask'] = LazyLoad(ToliasCadena2017Mask)
    # V1 properties benchmarks: orientation
    from .marques2020_ringach2002 import MarquesRingach2002V1CircularVariance, MarquesRingach2002V1Bandwidth, \
        MarquesRingach2002V1OrthogonalPreferredRatio, MarquesRingach2002V1OrientationSelective, \
        MarquesRingach2002V1CircularVarianceBandwidthRatio, \
        MarquesRingach2002V1OrthogonalPrefferredRatioCircularVarianceDifference
    from .marques2020_devalois1982a import MarquesDeValois1982V1PreferredOrientation
    pool['dicarlo.Marques2020_Ringach2002-circular_variance'] = LazyLoad(MarquesRingach2002V1CircularVariance)
    pool['dicarlo.Marques2020_Ringach2002-or_bandwidth'] = LazyLoad(MarquesRingach2002V1Bandwidth)
    pool['dicarlo.Marques2020_Ringach2002-orth_pref_ratio'] = LazyLoad(MarquesRingach2002V1OrthogonalPreferredRatio)
    pool['dicarlo.Marques2020_Ringach2002-or_selective'] = LazyLoad(MarquesRingach2002V1OrientationSelective)
    pool['dicarlo.Marques2020_Ringach2002-cv_bandwidth_ratio'] = \
        LazyLoad(MarquesRingach2002V1CircularVarianceBandwidthRatio)
    pool['dicarlo.Marques2020_Ringach2002-opr_cv_diff'] = \
        LazyLoad(MarquesRingach2002V1OrthogonalPrefferredRatioCircularVarianceDifference)
    pool['dicarlo.Marques2020_DeValois1982-pref_or'] = LazyLoad(MarquesDeValois1982V1PreferredOrientation)
    # V1 properties benchmarks: spatial frequency
    from .marques2020_devalois1982b import MarquesDeValois1982V1PeakSpatialFrequency
    from .marques2020_schiller1976 import MarquesSchiller1976V1SpatialFrequencyBandwidth, \
        MarquesSchiller1976V1SpatialFrequencySelective
    pool['dicarlo.Marques2020_DeValois1982-peak_sf'] = LazyLoad(MarquesDeValois1982V1PeakSpatialFrequency)
    pool['dicarlo.Marques2020_Schiller1976-sf_selective'] = LazyLoad(MarquesSchiller1976V1SpatialFrequencySelective)
    pool['dicarlo.Marques2020_Schiller1976-sf_bandwidth'] = LazyLoad(MarquesSchiller1976V1SpatialFrequencyBandwidth)
    # V1 properties benchmarks: surround
    from .marques2020_cavanaugh2002a import MarquesCavanaugh2002V1GratingSummationField, \
        MarquesCavanaugh2002V1SurroundDiameter, MarquesCavanaugh2002V1SurroundSuppressionIndex
    pool['dicarlo.Marques2020_Cavanaugh2002-grating_summation_field'] = \
        LazyLoad(MarquesCavanaugh2002V1GratingSummationField)
    pool['dicarlo.Marques2020_Cavanaugh2002-surround_diameter'] = \
        LazyLoad(MarquesCavanaugh2002V1SurroundDiameter)
    pool['dicarlo.Marques2020_Cavanaugh2002-surround_suppression_index'] = \
        LazyLoad(MarquesCavanaugh2002V1SurroundSuppressionIndex)
    # V1 properties benchmarks: texture modulation
    from .marques2020_freemanZiemba2013 import MarquesFreemanZiemba2013V1TextureModulationIndex, \
        MarquesFreemanZiemba2013V1AbsoluteTextureModulationIndex
    pool['dicarlo.Marques2020_FreemanZiemba2013-texture_modulation_index'] = \
        LazyLoad(MarquesFreemanZiemba2013V1TextureModulationIndex)
    pool['dicarlo.Marques2020_FreemanZiemba2013-abs_texture_modulation_index'] = \
        LazyLoad(MarquesFreemanZiemba2013V1AbsoluteTextureModulationIndex)
    # V1 properties benchmarks: selectivity
    from .marques2020_freemanZiemba2013 import MarquesFreemanZiemba2013V1TextureSelectivity, \
        MarquesFreemanZiemba2013V1TextureSparseness, MarquesFreemanZiemba2013V1VarianceRatio
    pool['dicarlo.Marques2020_FreemanZiemba2013-texture_selectivity'] = \
        LazyLoad(MarquesFreemanZiemba2013V1TextureSelectivity)
    pool['dicarlo.Marques2020_FreemanZiemba2013-texture_sparseness'] = \
        LazyLoad(MarquesFreemanZiemba2013V1TextureSparseness)
    pool['dicarlo.Marques2020_FreemanZiemba2013-texture_variance_ratio'] = \
        LazyLoad(MarquesFreemanZiemba2013V1VarianceRatio)
    # V1 properties benchmarks: magnitude
    from .marques2020_ringach2002 import MarquesRingach2002V1MaxDC, MarquesRingach2002V1ModulationRatio
    from .marques2020_freemanZiemba2013 import MarquesFreemanZiemba2013V1MaxTexture, MarquesFreemanZiemba2013V1MaxNoise
    pool['dicarlo.Marques2020_Ringach2002-max_dc'] = LazyLoad(MarquesRingach2002V1MaxDC)
    pool['dicarlo.Marques2020_Ringach2002-modulation_ratio'] = LazyLoad(MarquesRingach2002V1ModulationRatio)
    pool['dicarlo.Marques2020_FreemanZiemba2013-max_texture'] = LazyLoad(MarquesFreemanZiemba2013V1MaxTexture)
    pool['dicarlo.Marques2020_FreemanZiemba2013-max_noise'] = LazyLoad(MarquesFreemanZiemba2013V1MaxNoise)
    # Sanghavi2020 benchmarks
    from .sanghavi2020 import DicarloSanghavi2020V4PLS, DicarloSanghavi2020ITPLS
    pool['dicarlo.Sanghavi2020.V4-pls'] = LazyLoad(DicarloSanghavi2020V4PLS)
    pool['dicarlo.Sanghavi2020.IT-pls'] = LazyLoad(DicarloSanghavi2020ITPLS)
    from .sanghavijozwik2020 import DicarloSanghaviJozwik2020V4PLS, DicarloSanghaviJozwik2020ITPLS
    pool['dicarlo.SanghaviJozwik2020.V4-pls'] = LazyLoad(DicarloSanghaviJozwik2020V4PLS)
    pool['dicarlo.SanghaviJozwik2020.IT-pls'] = LazyLoad(DicarloSanghaviJozwik2020ITPLS)
    from .sanghavimurty2020 import DicarloSanghaviMurty2020V4PLS, DicarloSanghaviMurty2020ITPLS
    pool['dicarlo.SanghaviMurty2020.V4-pls'] = LazyLoad(DicarloSanghaviMurty2020V4PLS)
    pool['dicarlo.SanghaviMurty2020.IT-pls'] = LazyLoad(DicarloSanghaviMurty2020ITPLS)
    from .rajalingham2020 import DicarloRajalingham2020ITPLS
    pool['dicarlo.Rajalingham2020.IT-pls'] = LazyLoad(DicarloRajalingham2020ITPLS)

    return pool


def _public_benchmark_pool():
    """
    Benchmarks that are publicly usable, but are not used for the website.
    """
    pool = {}
    # neural benchmarks
    from .public_benchmarks import FreemanZiembaV1PublicBenchmark, FreemanZiembaV2PublicBenchmark, \
        MajajHongV4PublicBenchmark, MajajHongITPublicBenchmark
    pool['movshon.FreemanZiemba2013public.V1-pls'] = LazyLoad(FreemanZiembaV1PublicBenchmark)
    pool['movshon.FreemanZiemba2013public.V2-pls'] = LazyLoad(FreemanZiembaV2PublicBenchmark)
    pool['dicarlo.MajajHong2015public.V4-pls'] = LazyLoad(MajajHongV4PublicBenchmark)
    pool['dicarlo.MajajHong2015public.IT-pls'] = LazyLoad(MajajHongITPublicBenchmark)

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
evaluation_benchmark_pool = {**evaluation_benchmark_pool}
# provide unifying pool
benchmark_pool = {**public_benchmark_pool, **engineering_benchmark_pool,
                  **experimental_benchmark_pool, **evaluation_benchmark_pool}


@cache()
def load(name):
    if name not in benchmark_pool:
        raise ValueError(f"Unknown benchmark '{name}' - must choose from {list(benchmark_pool.keys())}")
    return benchmark_pool[name]
