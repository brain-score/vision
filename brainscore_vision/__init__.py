import logging
from typing import Dict, Any, Union, Callable

from brainio.assemblies import DataAssembly
from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Metric
from brainscore_core.plugin_management.import_plugin import import_plugin
from result_caching import store

from brainscore_vision.benchmarks import benchmark_pool
from brainscore_vision.model_interface import BrainModel

_logger = logging.getLogger(__name__)

data_registry: Dict[str, Callable[[], Union[DataAssembly, Any]]] = {}
""" Pool of available data """

metric_registry: Dict[str, Callable[[], Metric]] = {}
""" Pool of available metrics """

benchmark_registry: Dict[str, Callable[[], Benchmark]] = {}
""" Pool of available benchmarks """

model_registry: Dict[str, Callable[[], BrainModel]] = {}
""" Pool of available models """


def load_dataset(identifier: str) -> Union[DataAssembly, Any]:
    import_plugin('data', identifier)

    return data_registry[identifier]()


def load_metric(identifier: str, *args, **kwargs) -> Metric:
    import_plugin('metrics', identifier)

    return metric_registry[identifier](*args, **kwargs)


def load_benchmark(identifier: str) -> Benchmark:
    import_plugin('benchmarks', identifier)

    return benchmark_registry[identifier]()


def load_model(identifier: str) -> BrainModel:
    import_plugin('models', identifier)

    return model_registry[identifier]()


@store(identifier_ignore=['model', 'benchmark'])
def score_model(model_identifier, benchmark_identifier, model):
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
    benchmark = benchmark_pool[benchmark_identifier]
    _logger.debug("scoring model")
    score = benchmark(model)
    score.attrs['model_identifier'] = model_identifier
    score.attrs['benchmark_identifier'] = benchmark_identifier
    return score
