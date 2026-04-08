"""
Adapter wrapping legacy vision BrainModel to conform to UnifiedModel.

process() delegates to the legacy model's look_at(). The adapter is
the abstraction boundary: benchmarks above it call process(), legacy
code below it sees the same look_at() calls it always did.
"""

from typing import Any, Dict, Optional, Set

from brainscore_core.model_interface import UnifiedModel, TaskContext


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

    def process(self, stimuli, number_of_trials=1) -> Any:
        return self._legacy.look_at(stimuli, number_of_trials)

    def start_task(self, task_context: TaskContext) -> None:
        self._task_context = task_context
        # Legacy BrainModel.start_task(task, fitting_stimuli) takes two args
        self._legacy.start_task(task_context.task_type, task_context.fitting_stimuli)

    def start_recording(self, recording_target: str,
                        time_bins=None, recording_type=None, **kwargs) -> None:
        # Legacy BrainModel.start_recording(recording_target, time_bins)
        # Default time_bins for vision is [(70, 170)]
        self._legacy.start_recording(recording_target, time_bins or [(70, 170)])

    def visual_degrees(self):
        return self._legacy.visual_degrees()
