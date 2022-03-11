"""
The purpose of this file is to provide benchmarks based on publicly accessible data that can be run on candidate models
without restrictions. As opposed to the private benchmarks hosted on www.Brain-Score.org, models can be evaluated
without having to submit them to the online platform.
This allows for quick local prototyping, layer commitment, etc.
For the final model evaluation, candidate models should still be sent to www.Brain-Score.org to evaluate them on
held-out private data.
"""
import functools
import logging

import boto3
import brainio
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from brainio.fetch import BotoFetcher

from brainscore.benchmarks._neural_common import NeuralBenchmark
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.image_level_behavior import I2n
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.utils import LazyLoad
from .freemanziemba2013 import load_assembly as load_freemanziemba2013, VISUAL_DEGREES as freemanziemba2013_degrees, \
    NUMBER_OF_TRIALS as freemanziemba2013_trials, BIBTEX as freemanziemba2013_bibtex
from .kar2018 import load_assembly as load_kar2018, _DicarloKar2018
from .majajhong2015 import load_assembly as load_majajhong2015, VISUAL_DEGREES as majajhong2015_degrees, \
    NUMBER_OF_TRIALS as majajhong2015_trials, BIBTEX as majajhong2015_bibtex
from .rajalingham2018 import load_assembly as load_rajalingham2018, _DicarloRajalingham2018

_logger = logging.getLogger(__name__)


def _standard_benchmark(identifier, load_assembly, visual_degrees, number_of_trials, stratification_coord, bibtex):
    assembly_repetition = LazyLoad(lambda: load_assembly(average_repetitions=False))
    assembly = LazyLoad(lambda: load_assembly(average_repetitions=True))
    similarity_metric = CrossRegressedCorrelation(
        regression=pls_regression(), correlation=pearsonr_correlation(),
        crossvalidation_kwargs=dict(stratification_coord=stratification_coord))
    ceiler = InternalConsistency()
    return NeuralBenchmark(identifier=f"{identifier}-pls", version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=visual_degrees, number_of_trials=number_of_trials,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=None,
                           bibtex=bibtex)


def FreemanZiembaV1PublicBenchmark():
    return _standard_benchmark('movshon.FreemanZiemba2013.V1.public',
                               load_assembly=functools.partial(load_freemanziemba2013, region='V1', access='public'),
                               visual_degrees=freemanziemba2013_degrees, number_of_trials=freemanziemba2013_trials,
                               stratification_coord='texture_type', bibtex=freemanziemba2013_bibtex)


def FreemanZiembaV2PublicBenchmark():
    return _standard_benchmark('movshon.FreemanZiemba2013.V2.public',
                               load_assembly=functools.partial(load_freemanziemba2013, region='V2', access='public'),
                               visual_degrees=freemanziemba2013_degrees, number_of_trials=freemanziemba2013_trials,
                               stratification_coord='texture_type', bibtex=freemanziemba2013_bibtex)


def MajajHongV4PublicBenchmark():
    return _standard_benchmark('dicarlo.MajajHong2015.V4.public',
                               load_assembly=functools.partial(load_majajhong2015, region='V4', access='public'),
                               visual_degrees=majajhong2015_degrees, number_of_trials=majajhong2015_trials,
                               stratification_coord='object_name', bibtex=majajhong2015_bibtex)


def MajajHongITPublicBenchmark():
    return _standard_benchmark('dicarlo.MajajHong2015.IT.public',
                               load_assembly=functools.partial(load_majajhong2015, region='IT', access='public'),
                               visual_degrees=majajhong2015_degrees, number_of_trials=majajhong2015_trials,
                               stratification_coord='object_name', bibtex=majajhong2015_bibtex)


class Rajalingham2018MatchtosamplePublicBenchmark(_DicarloRajalingham2018):
    def __init__(self):
        super(Rajalingham2018MatchtosamplePublicBenchmark, self).__init__(metric=I2n(), metric_identifier='i2n')
        self._assembly = LazyLoad(lambda: load_rajalingham2018(access='public'))
        self._ceiling_func = lambda: self._metric.ceiling(self._assembly, skipna=True)


class Kar2018MatchtosamplePublicBenchmark(_DicarloKar2018):
    def __init__(self):
        super(Kar2018MatchtosamplePublicBenchmark, self).__init__(metric=I2n(), metric_identifier='i2n')
        self._assembly = LazyLoad(lambda: load_kar2018(access='public'))
        self._ceiling_func = lambda: self._metric.ceiling(self._assembly)


def list_public_assemblies():
    all_assemblies = brainio.list_assemblies()
    public_assemblies = []
    for assembly in all_assemblies:
        assy_model = brainio.lookup.lookup_assembly(assembly)
        if assy_model['location_type'] != 'S3':
            _logger.warning(f"Unknown location_type in assembly {assy_model}")
            continue
        probe_fetcher = _ProbeBotoFetcher(location=assy_model['location'], local_filename='probe')  # filename is unused
        if probe_fetcher.has_access():
            public_assemblies.append(assembly)
    return public_assemblies


class _ProbeBotoFetcher(BotoFetcher):
    def has_access(self):
        s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
        obj = s3.Object(self.bucketname, self.relative_path)
        try:
            # noinspection PyStatementEffect
            obj.content_length  # probe
            return True
        except ClientError:
            return False
