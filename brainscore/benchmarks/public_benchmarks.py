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
from .majajhong2015 import load_assembly as load_majajhong2015, VISUAL_DEGREES as majajhong2015_degrees
from .rajalingham2018 import load_assembly as load_rajalingham2018, DicarloRajalingham2018I2n

_logger = logging.getLogger(__name__)


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
                           bibtex= """@article {Majaj13402,
                                author = {Majaj, Najib J. and Hong, Ha and Solomon, Ethan A. and DiCarlo, James J.},
                                title = {Simple Learned Weighted Sums of Inferior Temporal Neuronal Firing Rates Accurately Predict Human Core Object Recognition Performance},
                                volume = {35},
                                number = {39},
                                pages = {13402--13418},
                                year = {2015},
                                doi = {10.1523/JNEUROSCI.5181-14.2015},
                                publisher = {Society for Neuroscience},
                                abstract = {To go beyond qualitative models of the biological substrate of object recognition, we ask: can a single ventral stream neuronal linking hypothesis quantitatively account for core object recognition performance over a broad range of tasks? We measured human performance in 64 object recognition tests using thousands of challenging images that explore shape similarity and identity preserving object variation. We then used multielectrode arrays to measure neuronal population responses to those same images in visual areas V4 and inferior temporal (IT) cortex of monkeys and simulated V1 population responses. We tested leading candidate linking hypotheses and control hypotheses, each postulating how ventral stream neuronal responses underlie object recognition behavior. Specifically, for each hypothesis, we computed the predicted performance on the 64 tests and compared it with the measured pattern of human performance. All tested hypotheses based on low- and mid-level visually evoked activity (pixels, V1, and V4) were very poor predictors of the human behavioral pattern. However, simple learned weighted sums of distributed average IT firing rates exactly predicted the behavioral pattern. More elaborate linking hypotheses relying on IT trial-by-trial correlational structure, finer IT temporal codes, or ones that strictly respect the known spatial substructures of IT ({\textquotedblleft}face patches{\textquotedblright}) did not improve predictive power. Although these results do not reject those more elaborate hypotheses, they suggest a simple, sufficient quantitative model: each object recognition task is learned from the spatially distributed mean firing rates (100 ms) of \~{}60,000 IT neurons and is executed as a simple weighted sum of those firing rates.SIGNIFICANCE STATEMENT We sought to go beyond qualitative models of visual object recognition and determine whether a single neuronal linking hypothesis can quantitatively account for core object recognition behavior. To achieve this, we designed a database of images for evaluating object recognition performance. We used multielectrode arrays to characterize hundreds of neurons in the visual ventral stream of nonhuman primates and measured the object recognition performance of \&gt;100 human observers. Remarkably, we found that simple learned weighted sums of firing rates of neurons in monkey inferior temporal (IT) cortex accurately predicted human performance. Although previous work led us to expect that IT would outperform V4, we were surprised by the quantitative precision with which simple IT-based linking hypotheses accounted for human behavior.},
                                issn = {0270-6474},
                                URL = {https://www.jneurosci.org/content/35/39/13402},
                                eprint = {https://www.jneurosci.org/content/35/39/13402.full.pdf},
                                journal = {Journal of Neuroscience}
                            }""")


def FreemanZiembaV1PublicBenchmark():
    return _standard_benchmark('movshon.FreemanZiemba2013.V1.public',
                               load_assembly=functools.partial(load_freemanziemba2013, region='V1', access='public'),
                               visual_degrees=freemanziemba2013_degrees, stratification_coord='texture_type')


def FreemanZiembaV2PublicBenchmark():
    return _standard_benchmark('movshon.FreemanZiemba2013.V2.public',
                               load_assembly=functools.partial(load_freemanziemba2013, region='V2', access='public'),
                               visual_degrees=freemanziemba2013_degrees, stratification_coord='texture_type')


def MajajHongV4PublicBenchmark():
    return _standard_benchmark('dicarlo.MajajHong2015.V4.public',
                               load_assembly=functools.partial(load_majajhong2015, region='V4', access='public'),
                               visual_degrees=majajhong2015_degrees, stratification_coord='object_name')


def MajajHongITPublicBenchmark():
    return _standard_benchmark('dicarlo.MajajHong2015.IT.public',
                               load_assembly=functools.partial(load_majajhong2015, region='IT', access='public'),
                               visual_degrees=majajhong2015_degrees, stratification_coord='object_name')


class RajalinghamMatchtosamplePublicBenchmark(DicarloRajalingham2018I2n):
    def __init__(self):
        super(RajalinghamMatchtosamplePublicBenchmark, self).__init__()
        self._assembly = LazyLoad(lambda: load_rajalingham2018(access='public'))
        self._ceiling_func = lambda: self._metric.ceiling(self._assembly, skipna=True)


def list_public_assemblies():
    all_assemblies = brainio_collection.list_assemblies()
    public_assemblies = []
    for assembly in all_assemblies:
        # https://github.com/brain-score/brainio_collection/blob/7892b9ec66c9e744766c794de4b73ebdf61d585c/brainio_collection/fetch.py#L181
        assy_model = brainio_collection.lookup.lookup_assembly(assembly)
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
