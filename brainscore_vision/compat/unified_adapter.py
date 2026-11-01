"""
Adapter wrapping legacy vision BrainModel to conform to the Subject interface
(formerly UnifiedModel).

process() delegates to the legacy model's look_at(). The adapter is
the abstraction boundary: benchmarks above it call process(), legacy
code below it sees the same look_at() calls it always did.
"""

from typing import Any, Dict, Optional, Set

from brainscore_core.model_interface import UnifiedModel, TaskContext
from brainscore_core.session_view import DeclaredInputSession, declared_only
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    _drain_stream_events,
    _stimuli_from_stream_session,
    measurement_settings,
)


class VisionModelAdapter(UnifiedModel):

    def __init__(self, legacy_model):
        self._legacy = legacy_model
        self._task_context: Optional[TaskContext] = None
        # Setup the legacy model currently has, so a repeated identical
        # configuration (a facade session re-sends it per measurement) is
        # applied once, as the legacy benchmark applied it.
        self._configured_task = None
        self._configured_recording = None

    @property
    def identifier(self) -> str:
        # Vision's identifier is a @property
        return self._legacy.identifier

    @property
    def region_layer_map(self) -> Dict[str, str]:
        if hasattr(self._legacy, 'layer_model') and hasattr(self._legacy.layer_model, 'region_layer_map'):
            return dict(self._legacy.layer_model.region_layer_map)
        return {}

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
        return super().out_channels | {'behavior'}

    def process(self, stimuli, number_of_trials=1, require_variance=False) -> Any:
        if require_variance:
            return self._legacy.look_at(stimuli, number_of_trials, require_variance=True)
        return self._legacy.look_at(stimuli, number_of_trials)

    def interact(self, session) -> None:
        """Run one session as the single legacy ``look_at`` it stands for.

        The session's task, recording target, time bins, trial count and
        variance request reach the legacy model through the same calls a
        legacy benchmark makes. Setup is applied when it changes, not on every
        measurement. Only declared input streams reach the legacy model.
        """
        session = DeclaredInputSession(session, self.in_channels)
        channels = tuple(session.requested_output_channels or ())
        if len(channels) != 1:
            raise ValueError('A legacy vision model answers one output channel '
                             f'per look_at; got {sorted(channels)}')
        channel, = channels
        if channel not in self.out_channels:
            raise ValueError(f'Undeclared output channel {channel!r}; declared: '
                             f'{sorted(self.out_channels)}')
        context = session.task_context
        settings = measurement_settings(session)
        if context is not None and context.task_type != 'passive':
            self.start_task(context)
        if channel.startswith('neural:'):
            self.start_recording(channel.split(':', 1)[1], time_bins=session.time_bins)
        events = _drain_stream_events(session)
        stimuli = _stimuli_from_stream_session(session, events)
        output = self.process(stimuli,
                              number_of_trials=settings.get('number_of_trials', 1),
                              require_variance=settings.get('require_variance', False))
        session.respond(StreamEvent(channel, output,
                                    events[-1].t_ms if events else None,
                                    meta={'driver': 'legacy'}))

    # Backwards-compatible: existing benchmarks call look_at() directly
    def look_at(self, stimuli, number_of_trials=1, require_variance=False):
        return self.process(stimuli, number_of_trials, require_variance)

    def start_task(self, task_context_or_task, fitting_stimuli=None,
                   **kwargs) -> None:
        # Accept both calling conventions:
        #   New: start_task(TaskContext(...)); fitting settings a legacy
        #        benchmark passed travel in metadata['task_settings']
        #   Legacy: start_task(task, fitting_stimuli, **settings)
        if isinstance(task_context_or_task, TaskContext):
            context = task_context_or_task
            task = context.task_type
            fitting = context.fitting_stimuli
            settings = dict(context.metadata.get('task_settings', {}))
        else:
            context = None
            task, fitting, settings = task_context_or_task, fitting_stimuli, dict(kwargs)
        key = (task, id(fitting), repr(sorted(settings.items())))
        self._task_context = context
        if key == self._configured_task:
            return
        self._legacy.start_task(task, declared_only(fitting, self.in_channels), **settings)
        self._configured_task = key

    def start_recording(self, recording_target: str,
                        time_bins=None, recording_type=None, **kwargs) -> None:
        # Legacy BrainModel.start_recording(recording_target, time_bins)
        # Default time_bins for vision is [(70, 170)]
        bins = time_bins or [(70, 170)]
        key = (repr(recording_target), repr(bins))
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
        self._task_context = None
        self._configured_task = None
        self._configured_recording = None

    @property
    def legacy_model(self):
        """The wrapped legacy model, for tools that need its own attributes.

        Unknown attributes are no longer forwarded implicitly: a benchmark
        sees only what the adapter declares (Contract Rules v3, rule 2.1).
        """
        return self._legacy
