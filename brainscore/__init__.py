import logging

# The following imports provide convenience methods.
# noinspection PyUnresolvedReferences
from brainio_collection import list_stimulus_sets, list_assemblies
# noinspection PyUnresolvedReferences
from brainio_collection.fetch import get_assembly as brainio_get_assembly, get_stimulus_set
from brainscore.benchmarks import benchmark_pool
from result_caching import store

_logger = logging.getLogger(__name__)


def get_assembly(name):
    assembly = brainio_get_assembly(name)
    if not hasattr(assembly.stimulus_set, 'name'):
        assembly.stimulus_set.name = assembly.stimulus_set_name

    stimulus_set_degrees = {'dicarlo.hvm': 8, 'movshon.FreemanZiemba2013': 4}
    if assembly.stimulus_set.name in stimulus_set_degrees:
        assembly.stimulus_set['degrees'] = stimulus_set_degrees[assembly.stimulus_set.name]
    return assembly


@store(identifier_ignore=['model', 'benchmark'])
def score_model(model_identifier, benchmark_identifier, model, benchmark=None):
    assert model is not None
    if benchmark is None:
        _logger.debug("retrieving benchmark")
        benchmark = benchmark_pool[benchmark_identifier]
    _logger.debug("scoring model")
    score = benchmark(model)
    return score
