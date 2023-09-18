import logging
from typing import Dict, Any, Union, Callable

from brainio.assemblies import DataAssembly
from brainio.stimuli import StimulusSet
from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Metric, Score
from brainscore_core.plugin_management.conda_score import wrap_score
from brainscore_core.plugin_management.import_plugin import import_plugin
from brainscore_vision.model_interface import BrainModel

_logger = logging.getLogger(__name__)

data_registry: Dict[str, Callable[[], Union[DataAssembly, Any]]] = {}
""" Pool of available data """

stimulus_set_registry: Dict[str, Callable[[], StimulusSet]] = {}
""" Pool of available stimulus sets"""

metric_registry: Dict[str, Callable[[], Metric]] = {}
""" Pool of available metrics """

benchmark_registry: Dict[str, Callable[[], Benchmark]] = {}
""" Pool of available benchmarks """

model_registry: Dict[str, Callable[[], BrainModel]] = {}
""" Pool of available models """


def load_dataset(identifier: str) -> Union[DataAssembly, Any]:
    import_plugin('brainscore_vision', 'data', identifier)

    return data_registry[identifier]()


def load_stimulus_set(identifier: str) -> Any:
    # dataset and stimulus_set are kept in the same directory
    import_plugin('brainscore_vision', 'data', identifier, registry_prefix='stimulus_set')

    return stimulus_set_registry[identifier]()


def load_metric(identifier: str, *args, **kwargs) -> Metric:
    import_plugin('brainscore_vision', 'metrics', identifier)

    return metric_registry[identifier](*args, **kwargs)


def load_benchmark(identifier: str) -> Benchmark:
    import_plugin('brainscore_vision', 'benchmarks', identifier)

    return benchmark_registry[identifier]()


def load_model(identifier: str) -> BrainModel:
    import_plugin('brainscore_vision', 'models', identifier)

    return model_registry[identifier]()


def _run_score(model_identifier: str, benchmark_identifier: str) -> Score:
    """
    Score the model referenced by the `model_identifier` on the benchmark referenced by the `benchmark_identifier`.
    """
    model: BrainModel = load_model(model_identifier)
    benchmark: Benchmark = load_benchmark(benchmark_identifier)
    score: Score = benchmark(model)
    score.attrs['model_identifier'] = model_identifier
    score.attrs['benchmark_identifier'] = benchmark_identifier
    return score


def score(model_identifier: str, benchmark_identifier: str, conda_active: bool=False) -> Score:
    """
    Score the model referenced by the `model_identifier` on the benchmark referenced by the `benchmark_identifier`.
    The model needs to implement the :class:`~brainscore.model_interface.BrainModel` interface
    so that the benchmark can interact with it.
    The benchmark will be looked up from the :data:`~brainscore_vision.benchmarks` and evaluates the model
    (looked up from :data:`~brainscore_vision.models`) on how brain-like it is under that benchmark's
    experimental paradigm, primate measurements, comparison metric, and ceiling.
    This results in a quantitative
    `Score <https://brain-score-core.readthedocs.io/en/latest/modules/metrics.html#brainscore_core.metrics.Score>`_
    ranging from 0 (least brain-like) to 1 (most brain-like under this benchmark).

    :param model_identifier: the identifier for the model
    :param benchmark_identifier: the identifier for the benchmark to test the model against
    :return: a Score of how brain-like the candidate model is under this benchmark. The score is normalized by
        this benchmark's ceiling such that 1 means the model matches the data to ceiling level.
    """
    return wrap_score(__file__,
                      model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                      score_function=_run_score, conda_active=conda_active )
