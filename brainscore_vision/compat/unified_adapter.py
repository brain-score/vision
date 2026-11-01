"""
Adapter wrapping legacy vision BrainModel to conform to the Subject interface
(formerly UnifiedModel).

process() delegates to the legacy model's look_at(). The adapter is
the abstraction boundary: benchmarks above it call process(), legacy
code below it sees the same look_at() calls it always did.
"""

from typing import Any, Dict, Optional, Set

from brainscore_core.model_interface import UnifiedModel, TaskContext
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    _drain_stream_events,
    _stimuli_from_stream_session,
)


class VisionModelAdapter(UnifiedModel):

    def __init__(self, legacy_model):
        self._legacy = legacy_model
        self._task_context: Optional[TaskContext] = None

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
        legacy benchmark makes, in the same order, exactly once.
        """
        channels = tuple(session.requested_output_channels or ())
        if len(channels) != 1:
            raise ValueError('A legacy vision model answers one output channel '
                             f'per look_at; got {sorted(channels)}')
        channel, = channels
        if channel not in self.out_channels:
            raise ValueError(f'Undeclared output channel {channel!r}; declared: '
                             f'{sorted(self.out_channels)}')
        context = session.task_context
        settings = dict(getattr(context, 'metadata', None) or {})
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
                                    events[-1].t_ms if events else 0.0,
                                    meta={'driver': 'legacy'}))

    # Backwards-compatible: existing benchmarks call look_at() directly
    def look_at(self, stimuli, number_of_trials=1, require_variance=False):
        return self.process(stimuli, number_of_trials, require_variance)

    def start_task(self, task_context_or_task, fitting_stimuli=None,
                   **kwargs) -> None:
        # Accept both calling conventions:
        #   New: start_task(TaskContext(...))
        #   Legacy: start_task(task, fitting_stimuli)
        if isinstance(task_context_or_task, TaskContext):
            self._task_context = task_context_or_task
            self._legacy.start_task(task_context_or_task.task_type,
                                    task_context_or_task.fitting_stimuli)
        else:
            self._task_context = None
            self._legacy.start_task(task_context_or_task, fitting_stimuli,
                                    **kwargs)

    def start_recording(self, recording_target: str,
                        time_bins=None, recording_type=None, **kwargs) -> None:
        # Legacy BrainModel.start_recording(recording_target, time_bins)
        # Default time_bins for vision is [(70, 170)]
        self._legacy.start_recording(recording_target, time_bins or [(70, 170)])

    def visual_degrees(self):
        return self._legacy.visual_degrees()

    def reset(self) -> None:
        reset = getattr(self._legacy, 'reset', None)
        if callable(reset):
            reset()
        self._task_context = None

    @property
    def legacy_model(self):
        """The wrapped legacy model, for tools that need its own attributes.

        Unknown attributes are no longer forwarded implicitly: a benchmark
        sees only what the adapter declares (Contract Rules v3, rule 2.1).
        """
        return self._legacy
