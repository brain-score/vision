from enum import Enum
from typing import List, Tuple


class BrainModel:
    RecordingTarget = Enum('RecordingTarget', " ".join(['V1', 'V2', 'V4', 'IT']))
    Task = Enum('Task', " ".join(['passive', 'probabilities', 'label', 'visual_search_obj_arr']))

    def visual_degrees(self) -> int:
        """
        The visual degrees this model covers as a single scalar.
        :return: e.g. `8`, or `10`
        """
        raise NotImplementedError()

    def look_at(self, stimuli):
        raise NotImplementedError()

    def start_task(self, task: Task, fitting_stimuli):
        raise NotImplementedError()

    def start_recording(self, recording_target: RecordingTarget, time_bins=List[Tuple[int]]):
        raise NotImplementedError()
