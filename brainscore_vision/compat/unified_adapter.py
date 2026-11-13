"""
Adapter wrapping legacy vision BrainModel to conform to the Subject interface
(formerly UnifiedModel).

process() delegates to the legacy model's look_at(). The adapter is
the abstraction boundary: benchmarks above it call process(), legacy
code below it sees the same look_at() calls it always did.
"""

from typing import Any, Dict, List, Optional, Set

from brainscore_core.model_interface import UnifiedModel, TaskContext
from brainscore_core.session_view import declared_only
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    drain_session_inputs,
    measurement_settings,
    session_stimuli,
)


class VisionModelAdapter(UnifiedModel):

    def __init__(self, legacy_model):
        self._legacy = legacy_model
        self._task: Optional[TaskContext] = None
        # Setup the legacy model currently has, so a repeated identical
        # configuration (a facade session re-sends it per measurement) is
        # applied once, as the legacy benchmark applied it.
        self._configured_task = None
        self._configured_recording = None

    @property
    def identifier(self) -> str:
        # Vision's identifier is a @property
        return self._legacy.identifier

    def _legacy_layer_map(self):
        if hasattr(self._legacy, 'layer_model') and hasattr(self._legacy.layer_model, 'region_layer_map'):
            return self._legacy.layer_model.region_layer_map
        return {}

    @property
    def region_layer_map(self) -> Dict[str, str]:
        """The region -> layer commitments the legacy model has made so far."""
        return dict(self._legacy_layer_map())

    def _mappable_regions(self) -> List[str]:
        """Every region the legacy model is wired to map, without a layer search.

        The committed regions come first. A lazy map (a ``ModelCommitment``
        with no ``region_layer_map/<id>.json`` and no ``region_layer_map=``)
        also declares, in its public ``region_benchmarks``, the regions its
        layer selection can search. The search itself runs where it always ran:
        at the first legacy ``look_at`` of a region.
        """
        layer_map = self._legacy_layer_map()
        regions = list(layer_map)  # dict iteration: committed regions only, no search
        # Read as a public declaration; a plain committed dict has none.
        searchable = getattr(layer_map, 'region_benchmarks', None) or {}
        regions += [region for region in searchable if region not in regions]
        return regions

    @property
    def probe_safe_regions(self) -> List[str]:
        """Regions with a committed layer: recording them starts no layer search.

        The memory pre-flight probes only these regions on a benchmark that
        does not name the region it measures.
        """
        return list(self._legacy_layer_map())

    @property
    def supported_modalities(self) -> Set[str]:
        return {'vision'}

    @property
    def required_modalities(self) -> Set[str]:
        # Legacy vision models are unimodal — they can only process images.
        # Hard-require vision so pre-flight rejects pairings against a
        # benchmark that does not provide vision stimuli.
        return {'vision'}

    @property
    def out_channels(self) -> Set[str]:
        # A legacy BrainModel answers behavioral tasks as well as recording.
        # Neural channels come from the regions the legacy model can map, so a
        # lazy layer search declares the same channels before and after it runs.
        return {f'neural:{region}' for region in self._mappable_regions()} | {'behavior'}

    def process(self, stimuli, number_of_trials=1, require_variance=False) -> Any:
        if require_variance:
            return self._legacy.look_at(stimuli, number_of_trials, require_variance=True)
        return self._legacy.look_at(stimuli, number_of_trials)

    def participate(self, session) -> None:
        """Run one session as the legacy ``look_at`` calls it stands for.

        The session's task, recording target, time bins, trial count and
        variance request reach the legacy model through the same calls a
        legacy benchmark makes. Setup is applied when it changes, not on every
        measurement. Only declared input streams reach the legacy model.
        A session that requests several ``neural:<region>`` channels gets, for
        each region in request order, the ``start_recording`` and ``look_at``
        that a session for that region alone gets.
        """
        session = session.declare_inputs(self.in_channels)
        channels = tuple(session.outputs or ())
        if not channels or (len(channels) > 1
                            and not all(c.startswith('neural:') for c in channels)):
            raise ValueError('A legacy vision model answers one output channel per '
                             f'look_at, or several neural:<region> channels; got {sorted(channels)}')
        undeclared = [channel for channel in channels if channel not in self.out_channels]
        if undeclared:
            raise ValueError(f'Undeclared output channel {undeclared[0]!r}; declared: '
                             f'{sorted(self.out_channels)}')
        task = session.task
        settings = measurement_settings(session)
        if task is not None and task.task_type != 'passive':
            self.start_task(task)
        elif self._configured_task is not None and self._configured_task[0] != 'passive':
            # An earlier session left the legacy model answering a task. A
            # passive session leaves it the legacy way, as a legacy benchmark
            # does with start_task('passive'). A fresh subject gets no call.
            self.start_task('passive')
        events = drain_session_inputs(session)
        stimuli = session_stimuli(session, events)
        for channel in channels:
            if channel.startswith('neural:'):
                self.start_recording(channel.split(':', 1)[1], time_bins=session.time_bins)
            output = self.process(stimuli,
                                  number_of_trials=settings.get('number_of_trials', 1),
                                  require_variance=settings.get('require_variance', False))
            session.respond(StreamEvent(channel, output,
                                        events[-1].t_ms if events else None,
                                        meta={'driver': 'legacy'}))

    # Backwards-compatible: existing benchmarks call look_at() directly
    def look_at(self, stimuli, number_of_trials=1, require_variance=False):
        return self.process(stimuli, number_of_trials, require_variance)

    def start_task(self, task_or_type, fitting_stimuli=None,
                   **kwargs) -> None:
        # Accept both calling conventions:
        #   New: start_task(TaskContext(...)); fitting settings a legacy
        #        benchmark passed travel in metadata['task_settings']
        #   Legacy: start_task(task, fitting_stimuli, **settings)
        if isinstance(task_or_type, TaskContext):
            task = task_or_type
            task_type = task.task_type
            fitting = task.fitting_stimuli
            settings = dict(task.metadata.get('task_settings', {}))
        else:
            task = None
            task_type, fitting, settings = task_or_type, fitting_stimuli, dict(kwargs)
        key = (task_type, id(fitting), repr(sorted(settings.items())))
        self._task = task
        if key == self._configured_task:
            return
        # Forward what the benchmark passed: a task without fitting data
        # (Hebart2023 odd_one_out) reaches a start_task(self, task) unchanged.
        if fitting is None:
            self._legacy.start_task(task_type, **settings)
        else:
            self._legacy.start_task(task_type, declared_only(fitting, self.in_channels), **settings)
        self._configured_task = key

    def start_recording(self, recording_target: str,
                        time_bins=None, recording_type=None, **kwargs) -> None:
        # Legacy BrainModel.start_recording(recording_target, time_bins)
        # Default time_bins for vision is [(70, 170)]
        # Explicit check: an ndarray of bins (temporal benchmarks) has no truth value.
        bins = [(70, 170)] if time_bins is None or len(time_bins) == 0 else time_bins
        key = (repr(recording_target), _time_bins_key(bins))
        if key == self._configured_recording:
            return
        self._legacy.start_recording(recording_target, bins)
        self._configured_recording = key

    def visual_degrees(self):
        return self._legacy.visual_degrees()

    def reset(self) -> None:
        reset = getattr(self._legacy, 'reset', None)
        if callable(reset):
            reset()
        self._task = None
        self._configured_task = None
        self._configured_recording = None

    @property
    def legacy_model(self):
        """The wrapped legacy model, for tools that need its own attributes.

        Unknown attributes are no longer forwarded implicitly: a benchmark
        sees only what the adapter declares (Contract Rules v3, rule 2.1).
        """
        return self._legacy


def _time_bins_key(bins):
    """An exact setup key for time bins.

    Lists and tuples keep their ``repr`` key. An ndarray (temporal benchmarks)
    gets a structural key that does not depend on NumPy display precision.
    """
    import numpy as np
    if not isinstance(bins, np.ndarray):
        return repr(bins)
    return _exact(bins)


def _exact(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return ('ndarray', 'O', value.shape, tuple(_exact(item) for item in value.ravel()))
        return ('ndarray', value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_exact(item) for item in value))
    if isinstance(value, np.generic):
        return ('scalar', value.dtype.str, value.tobytes())
    return (type(value).__name__, value)
