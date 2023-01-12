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

from brainscore_vision.benchmark_helpers._neural_common import NeuralBenchmark
from brainscore_vision.metrics.ceiling import InternalConsistency
from brainscore_vision.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore_vision.utils import LazyLoad
from .benchmark import load_assembly as load_freemanziemba2013, NUMBER_OF_TRIALS as freemanziemba2013_trials, \
    VISUAL_DEGREES as freemanziemba2013_degrees, BIBTEX as freemanziemba2013_bibtex

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



