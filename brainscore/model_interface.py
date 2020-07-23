from enum import Enum

from brainio_base.stimuli import StimulusSet
from typing import List, Tuple, Union


class BrainModel:
    """
    The BrainModel interface defines an API for models to follow.
    Benchmarks will use this interface to treat models as an experimental subject
    without needing to know about the details of the model implementation.
    """

    RecordingTarget = Enum('RecordingTarget', " ".join(['V1', 'V2', 'V4', 'IT']))
    """
    location to record from
    """

    Task = Enum('Task', " ".join(['passive', 'probabilities', 'label']))
    """
    task to perform
    """

    def visual_degrees(self) -> int:
        """
        The visual degrees this model covers as a single scalar.

        :return: e.g. `8`, or `10`
        """
        raise NotImplementedError()

    def look_at(self, stimuli: Union[StimulusSet, List[str]]):
        """
        Digest a set of stimuli and return requested outputs. Which outputs to return is instructed by the
        :meth:`~brainscore.model_interface.BrainMode.start_task` and
        :meth:`~brainscore.model_interface.BrainModel.start_recording` methods.

        :param stimuli: A set of stimuli, passed as either a :class:`~brainio_base.stimuli.StimulusSet`
            or a list of image file paths
        :return: recordings or task behaviors as instructed
        """
        raise NotImplementedError()

    def start_task(self, task: Task, fitting_stimuli):
        """
        Instructs the model to begin one of the tasks specified in :data:`~brainscore.model_interface.BrainModel.Task`.
        For all followings call of :meth:`~brainscore.model_interface.BrainModel.look_at`, the model returns the
        expected outputs for the specified task.

        :param task: The task the model should perform, and thus which outputs it should return
        :param fitting_stimuli: A set of stimuli for the model to learn on, e.g. image-label pairs
        """
        raise NotImplementedError()

    def start_recording(self, recording_target: RecordingTarget, time_bins=List[Tuple[int]]):
        """
        Instructs the model to begin recording in a specified
        :data:`~brainscore.model_interface.BrainModel.RecordingTarget` and return the specified `time_bins`.
        For all followings call of :meth:`~brainscore.model_interface.BrainModel.look_at`, the model returns the
        corresponding recordings. These recordings are a :class:`~brainio_base.assemblies.NeuroidAssembly` with exactly
        3 dimensions:

        - `presentation`: the presented stimuli (cf. stimuli argument of
          :meth:`~brainscore.model_interface.BrainModel.look_at`). If a :class:`~brainio_base.stimuli.StimulusSet`
          was passed, the recordings should contain all of the :class:`~brainio_base.stimuli.StimulusSet` columns as
          coordinates on this dimension. The `image_id` coordinate is required in either case.
        - `neuroid`: the recorded neuroids (neurons or mixtures thereof). They should all be part of the
          specified :data:`~brainscore.model_interface.BrainModel.RecordingTarget`. The coordinates of this
          dimension should again include as much information as is available, at the very least a `neuroid_id`.
        - `time_bins`: the time bins of each recording slice. This dimension should contain at least 2 coordinates:
          `time_bin_start` and `time_bin_end`, where one `time_bin` is the bin between start and end.
          For instance, a 70-170ms time_bin would be marked as `time_bin_start=70` and `time_bin_end=170`.
          If only one time_bin is requested, the model may choose to omit this dimension.

        :param recording_target: which location to record from
        :param time_bins: which time_bins to record as a list of integer tuples,
            e.g. `[(50, 100), (100, 150), (150, 200)]` or `[(70, 170)]`
        """
        raise NotImplementedError()
