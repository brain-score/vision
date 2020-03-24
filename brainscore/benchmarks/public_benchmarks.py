"""
The purpose of this file is to provide benchmarks based on publicly accessible data that can be run on candidate models
without restrictions. As opposed to the private benchmarks hosted on www.Brain-Score.org, models can be evaluated
without having to submit them to the online platform.
This allows for quick local prototyping, layer commitment, etc.
For the final model evaluation, candidate models should still be sent to www.Brain-Score.org to evaluate them on
held-out private data.
"""
import boto3
import functools
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

import brainio_collection
from brainio_collection.fetch import BotoFetcher
from brainscore.benchmarks._neural_common import NeuralBenchmark
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.utils import LazyLoad
from .freemanziemba2013 import load_assembly as load_freemanziemba2013, VISUAL_DEGREES as freemanziemba2013_degrees
from .majaj2015 import load_assembly as load_majaj2015, VISUAL_DEGREES as majaj2015_degrees
from .rajalingham2018 import load_assembly as load_rajalingham2018, DicarloRajalingham2018I2n


def _standard_benchmark(identifier, load_assembly, visual_degrees, stratification_coord):
    assembly_repetition = LazyLoad(lambda: load_assembly(average_repetitions=False))
    assembly = LazyLoad(lambda: load_assembly(average_repetitions=True))
    similarity_metric = CrossRegressedCorrelation(
        regression=pls_regression(), correlation=pearsonr_correlation(),
        crossvalidation_kwargs=dict(stratification_coord=stratification_coord))
    ceiler = InternalConsistency()
    return NeuralBenchmark(identifier=f"{identifier}-pls", version=1,
                           assembly=assembly, similarity_metric=similarity_metric, visual_degrees=visual_degrees,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=None, paper_link='http://www.jneurosci.org/content/35/39/13402.short')


def FreemanZiembaV1PublicBenchmark():
    return _standard_benchmark('movshon.FreemanZiemba2013.V1.public',
                               load_assembly=functools.partial(load_freemanziemba2013, region='V1', access='public'),
                               visual_degrees=freemanziemba2013_degrees, stratification_coord='texture_type')


def FreemanZiembaV2PublicBenchmark():
    return _standard_benchmark('movshon.FreemanZiemba2013.V2.public',
                               load_assembly=functools.partial(load_freemanziemba2013, region='V2', access='public'),
                               visual_degrees=freemanziemba2013_degrees, stratification_coord='texture_type')


def MajajV4PublicBenchmark():
    return _standard_benchmark('dicarlo.Majaj2015.V4.public',
                               load_assembly=functools.partial(load_majaj2015, region='V4', access='public'),
                               visual_degrees=majaj2015_degrees, stratification_coord='object_name')


def MajajITPublicBenchmark():
    return _standard_benchmark('dicarlo.Majaj2015.IT.public',
                               load_assembly=functools.partial(load_majaj2015, region='IT', access='public'),
                               visual_degrees=majaj2015_degrees, stratification_coord='object_name')


class RajalinghamMatchtosamplePublicBenchmark(DicarloRajalingham2018I2n):
    def __init__(self):
        super(RajalinghamMatchtosamplePublicBenchmark, self).__init__()
        self._assembly = LazyLoad(lambda: load_rajalingham2018(access='public'))


def list_public_assemblies():
    all_assemblies = brainio_collection.list_assemblies()
    public_assemblies = []
    for assembly in all_assemblies:
        access = True
        # https://github.com/brain-score/brainio_collection/blob/a7a1eed2afafa0988d2b9da76091b3f61942e4d1/brainio_collection/fetch.py#L208
        assy_model = brainio_collection.assemblies.lookup_assembly(assembly)
        for store_map in assy_model.assembly_store_maps:
            probe_fetcher = _ProbeBotoFetcher(location=store_map.assembly_store_model.location,
                                              unique_name=store_map.assembly_store_model.unique_name)
            if not probe_fetcher.has_access():
                access = False
                break
        if access:
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
