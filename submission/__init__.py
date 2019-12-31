import logging

from brainscore.benchmarks import benchmark_pool
# from brainscore.assemblies.private import load_assembly as load_private_assembly
# from brainscore.assemblies.public import load_assembly as load_public_assembly
from brainscore.public_benchmarks import list_public_assemblies, _standard_benchmark
from result_caching import store

_logger = logging.getLogger(__name__)


@store(identifier_ignore=['model', 'benchmark'])
def map_and_score_model(model_identifier, benchmark_identifier, model, benchmark=None):
    assert model is not None
    if benchmark is None:
        _logger.debug("retrieving benchmark")
        benchmark = benchmark_pool[benchmark_identifier]
    _logger.debug("mapping model")
    mapped_model = model.map(benchmark.training_benchmark, benchmark.validation_benchmark)
    _logger.debug("scoring mapped model")
    score = benchmark(mapped_model)
    return score


@store(identifier_ignore=['model', 'benchmark'])
def score_model(model_identifier, benchmark_identifier, model, benchmark=None):
    assert model is not None
    if benchmark is None:
        _logger.debug("retrieving benchmark")
        benchmark = benchmark_pool[benchmark_identifier]
    _logger.debug("scoring model")
    score = benchmark(model)
    return score


def score_layers(model_identifier, benchmark_identifier, model, layers, benchmark=None):
    assert model is not None
    if benchmark is None:
        _logger.debug("retrieving benchmark")
        benchmark = benchmark_pool[benchmark_identifier]
    from model_tools.brain_transformation import LayerScores
    scorer = LayerScores(model_identifier=model_identifier, activations_model=model)
    scores = scorer(benchmark=benchmark, benchmark_identifier=benchmark_identifier, layers=layers)
    return scores
