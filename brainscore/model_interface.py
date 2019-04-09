from enum import Enum


class BrainModel:
    RecordingTarget = Enum('RecordingTarget', " ".join(['V1', 'V2', 'V4', 'IT']))
    Task = Enum('Task', " ".join(['passive', 'probabilities']))

    def look_at(self, stimuli):
        raise NotImplementedError()

    def start_task(self, task: Task):
        raise NotImplementedError()

    def start_recording(self, recording_target: RecordingTarget):
        raise NotImplementedError()
