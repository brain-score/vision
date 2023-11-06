"""
The :class:`~brainscore.model_interface.BrainModel` interface is the central communication point
between benchmarks and models.
"""

from typing import List, Tuple, Union

from brainio.stimuli import StimulusSet


class BrainModel:
    """
    The BrainModel interface defines an API for models to follow.
    Benchmarks will use this interface to treat models as an experimental subject
    without needing to know about the details of the model implementation.
    """

    @property
    def identifier(self) -> str:
        """
        The unique identifier for this model.

        :return: e.g. `'CORnet-S'`, or `'alexnet'`
        """
        raise NotImplementedError()

    def visual_degrees(self) -> int:
        """
        The visual degrees this model covers as a single scalar.

        :return: e.g. `8`, or `10`
        """
        raise NotImplementedError()

    class Task:
        """ task to perform """

        passive = 'passive'
        """
        Passive fixation, i.e. do not perform any task, but fixate on the center of the screen.
        Does not output anything, but can be useful to fully specify the experimental setup.

        Example:

        Setting up passive fixation with `start_task(BrainModel.Task.passive)` and calling `look_at(...)` could output

        .. code-block:: python

           None
        """

        label = 'label'
        """ 
        Predict the label for each stimulus. 
        Output a :class:`~brainio.assemblies.BehavioralAssembly` with labels as the values.

        The labeling domain can be specified in the second argument, e.g. `'imagenet'` for 1,000 ImageNet synsets, 
        or an explicit list of label strings. The model choices must be part of the labeling domain.

        Example:

        Setting up a labeling task for ImageNet synsets with `start_task(BrainModel.Task.label, 'imagenet')` 
        and calling `look_at(...)` could output 

        .. code-block:: python

           <xarray.BehavioralAssembly (presentation: 3, choice: 1)>
                array([['n02107574'], ['n02123045'], ['n02804414']]), # the ImageNet synsets
                Coordinates:
                  * presentation  (presentation) MultiIndex
                  - stimulus_id   (presentation) object 'hash1' 'hash2' 'hash3'
                  - stimulus_path (presentation) object '/home/me/.brainio/demo_stimuli/image1.png' ...
                  - logit         (presentation) int64 239 282 432
                  - synset        (presentation) object 'n02107574' 'n02123045' 'n02804414'

        Example:

        Setting up a labeling task for 2 custom labels with `start_task(BrainModel.Task.label, ['dog', 'cat'])` 
        and calling `look_at(...)` could output 

        .. code-block:: python

           <xarray.BehavioralAssembly (presentation: 3, choice: 1)>
                array([['dog'], ['cat'], ['cat']]), # the labels
                Coordinates:
                  * presentation  (presentation) MultiIndex
                  - stimulus_id   (presentation) object 'hash1' 'hash2' 'hash3'
                  - stimulus_path (presentation) object '/home/me/.brainio/demo_stimuli/image1.png' ...
        """

        probabilities = 'probabilities'
        """ 
        Predict the per-label probabilities for each stimulus. 
        Output a :class:`~brainio.assemblies.BehavioralAssembly` with probabilities as the values.

        The model must be supplied with `fitting_stimuli` in the second argument which allow it to train a readout 
        for a particular set of labels and image distribution. 
        The `fitting_stimuli` are a :class:`~brainio.stimuli.StimulusSet` and must include an `image_label` column 
        which is used as the labels to fit to.

        Example:

        Setting up a probabilities task `start_task(BrainModel.Task.probabilities, <fitting_stimuli>)` 
        (where `fitting_stimuli` includes 5 distinct labels)
        and calling `look_at(<test_stimuli>)` could output 

        .. code-block:: python

           <xarray.BehavioralAssembly (presentation: 3, choice: 5)>
                array([[0.9 0.1 0.0 0.0 0.0]
                       [0.0 0.0 0.8 0.0 0.2]
                       [0.0 0.0 0.0 1.0 0.0]]), # the probabilities
                Coordinates:
                  * presentation  (presentation) MultiIndex
                  - stimulus_id   (presentation) object 'hash1' 'hash2' 'hash3'
                  - stimulus_path (presentation) object '/home/me/.brainio/demo_stimuli/image1.png' ...
                  - choice        (choice) object 'dog' 'cat' 'chair' 'flower' 'plane'
        """

    def start_task(self, task: Task, fitting_stimuli):
        """
        Instructs the model to begin one of the tasks specified in :data:`~brainscore.model_interface.BrainModel.Task`.
        For all followings call of :meth:`~brainscore.model_interface.BrainModel.look_at`, the model returns the
        expected outputs for the specified task.

        :param task: The task the model should perform, and thus which outputs it should return
        :param fitting_stimuli: A set of stimuli for the model to learn on, e.g. image-label pairs
        """
        raise NotImplementedError()

    class RecordingTarget:
        """ location to record from """
        V1 = 'V1'
        V2 = 'V2'
        V4 = 'V4'
        IT = 'IT'

    class RegionMacaque(RecordingTarget):
        """ macaque location to record from """
        V1ventral = 'mV1v'
        V2ventral = 'mV2v'
        V4 = 'mV4'

    class RegionHuman(RecordingTarget):
        """ human location to record from """
        V1ventral = 'hV1v'
        V1dorsal = 'hV1d'
        V2ventral = 'hV2v'
        V2dorsal = 'hV2d'
        V3ventral = 'hV3v'
        V3dorsal = 'hV3d'
        V4 = 'hV4'
        whole_brain = 'whole_brain'

    def start_recording(self, recording_target: RecordingTarget, time_bins=List[Tuple[int]]):
        """
        Instructs the model to begin recording in a specified
        :data:`~brainscore.model_interface.BrainModel.RecordingTarget` and return the specified `time_bins`.
        For all followings call of :meth:`~brainscore.model_interface.BrainModel.look_at`, the model returns the
        corresponding recordings. These recordings are a :class:`~brainio.assemblies.NeuroidAssembly` with exactly
        3 dimensions:

        - `presentation`: the presented stimuli (cf. stimuli argument of
          :meth:`~brainscore.model_interface.BrainModel.look_at`). If a :class:`~brainio.stimuli.StimulusSet`
          was passed, the recordings should contain all of the :class:`~brainio.stimuli.StimulusSet` columns as
          coordinates on this dimension. The `stimulus_id` coordinate is required in either case.
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

    def look_at(self, stimuli: Union[StimulusSet, List[str]], number_of_trials=1):
        """
        Digest a set of stimuli and return requested outputs. Which outputs to return is instructed by the
        :meth:`~brainscore.model_interface.BrainMode.start_task` and
        :meth:`~brainscore.model_interface.BrainModel.start_recording` methods.

        :param stimuli: A set of stimuli, passed as either a :class:`~brainio.stimuli.StimulusSet`
            or a list of image file paths
        :param number_of_trials: The number of repeated trials of the stimuli that the model should average over.
            E.g. 10 or 35. Non-stochastic models can likely ignore this parameter.
        :return: recordings or task behaviors as instructed
        """
        raise NotImplementedError()
