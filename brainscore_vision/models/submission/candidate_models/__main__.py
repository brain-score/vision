from collections import OrderedDict

import argparse
import fire
import logging
import sys

from brainscore import score_model as score_model_function
from candidate_models import get_activations
from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import brain_translated_pool
from candidate_models.model_commitments.model_layer_def import model_layers

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
args, remaining_args = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level),
                    format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')
for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3', 'peewee', 'PIL']:
    logging.getLogger(disable_logger).setLevel(logging.WARNING)


def activations(model, stimulus_set, layers=None):
    logger.info(f"Retrieving activations for {model} layers {layers} on stimulus set {stimulus_set} with args {args}")
    if isinstance(layers, str):
        layers = [layers]
    model_implementation = base_model_pool[model]
    default_layers = model_layers[model]
    layers = layers or default_layers
    result = get_activations(model_implementation, layers=layers, stimulus_set=stimulus_set)
    print(result)


def score_model(model, benchmark):
    logger.info(f"Scoring {model} on benchmark {benchmark} with args {args}")
    _model = brain_translated_pool[model]
    result = score_model_function(model_identifier=model, benchmark_identifier=benchmark, model=_model)
    print(result)


def score_on_benchmarks(model, *benchmarks):
    scores = OrderedDict()
    for benchmark in benchmarks:
        _model = brain_translated_pool[model]
        score = score_model_function(model_identifier=model, benchmark_identifier=benchmark, model=_model)
        scores[benchmark] = score
    print(scores)


logger.info(f"Running {' '.join(sys.argv)}")
fire.Fire(command=remaining_args)
