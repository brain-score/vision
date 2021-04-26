import logging

# The following imports provide convenience methods.
# noinspection PyUnresolvedReferences
from brainio_collection import list_stimulus_sets, list_assemblies
# noinspection PyUnresolvedReferences
from brainio_collection.fetch import get_assembly as brainio_get_assembly, get_stimulus_set
from brainscore.benchmarks import benchmark_pool
from result_caching import store
from brainscore.tolerance import get_benchmark

_logger = logging.getLogger(__name__)


def get_assembly(name):
    assembly = brainio_get_assembly(name)
    assert hasattr(assembly.stimulus_set, 'identifier')
    assert assembly.stimulus_set.identifier == assembly.stimulus_set_identifier
    return assembly


@store(identifier_ignore=['model', 'benchmark'])
def score_model(model_identifier, benchmark_identifier, model, **kwargs):
    """
    Score a given model on a given benchmark.
    The model needs to implement the :class:`~brainscore.model_interface.BrainModel` interface so that the benchmark can
    interact with it.
    The benchmark will be looked up from the :data:`~brainscore.benchmarks.benchmark_pool` and evaluates the model
    on how brain-like it is under that benchmark's experimental paradigm, primate measurements, comparison metric, and
    ceiling. This results in a quantitative :class:`~brainscore.metrics.Score` ranging from 0 (least brain-like) to 1
    (most brain-like under this benchmark).

    The results of this method are cached by default (according to the identifiers), calling it twice with the same
    identifiers will only invoke once.

    :param model_identifier: a unique identifier for this model
    :param model: the model implementation following the :class:`~brainscore.model_interface.BrainModel` interface
    :param benchmark_identifier: the identifier of the benchmark to test the model against
    :return: a :class:`~brainscore.metrics.Score` of how brain-like the candidate model is under this benchmark. The
                score is normalized by this benchmark's ceiling such that 1 means the model matches the data to ceiling
                level.

    :seealso: :class:`brainscore.benchmarks.Benchmark`
    :seealso: :class:`brainscore.model_interface.BrainModel`
    """
    # model_identifier variable is not unused, the result caching component uses it to identify the cached results
    assert model is not None
    _logger.debug("retrieving benchmark")

    # Check if it's as special identifier from the tolerance project
    if benchmark_identifier.startswith('tol_'):
        benchmark = get_benchmark(benchmark_identifier, **kwargs)
    else:
        benchmark = benchmark_pool[benchmark_identifier]


    _logger.debug("scoring model")
    score = benchmark(model)
    score.attrs['model_identifier'] = model_identifier
    score.attrs['benchmark_identifier'] = benchmark_identifier
    return score
