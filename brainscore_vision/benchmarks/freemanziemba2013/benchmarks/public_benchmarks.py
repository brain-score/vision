"""
The purpose of this file is to provide benchmarks based on publicly accessible data that can be run on candidate models
without restrictions. As opposed to the private benchmarks hosted on www.Brain-Score.org, models can be evaluated
without having to submit them to the online platform.
This allows for quick local prototyping, layer commitment, etc.
For the final model evaluation, candidate models should still be sent to www.Brain-Score.org to evaluate them on
held-out private data.
"""
import logging

from brainscore_vision import load_metric, load_ceiling
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark
from brainscore_vision.utils import LazyLoad
from .benchmark import load_assembly, NUMBER_OF_TRIALS, VISUAL_DEGREES, BIBTEX

_logger = logging.getLogger(__name__)


def _freemanziemba2013_public_benchmark(region: str):
    assembly_repetition = LazyLoad(lambda: load_assembly(region=region, access='public', average_repetitions=False))
    assembly = LazyLoad(lambda: load_assembly(region=region, access='public', average_repetitions=True))
    similarity_metric = load_metric('pls', crossvalidation_kwargs=dict(stratification_coord='texture_type'))
    ceiler = load_ceiling('internal_consistency')
    return NeuralBenchmark(identifier=f"FreemanZiemba2013.{region}.public-pls", version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=None,
                           bibtex=BIBTEX)


def FreemanZiembaV1PublicBenchmark():
    return _freemanziemba2013_public_benchmark('V1')


def FreemanZiembaV2PublicBenchmark():
    return _freemanziemba2013_public_benchmark('V2')
