#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 23:15:17 2024

@author: costantino_ai
"""
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.model_interface import BrainModel
from brainscore_vision import load_stimulus_set, load_metric, load_dataset
from brainscore_vision.utils import LazyLoad


BIBTEX = """@article {Maniquet2024.04.02.587669,
	author = {Maniquet, Tim and de Beeck, Hans Op and Costantino, Andrea Ivan},
	title = {Recurrent issues with deep neural network models of visual recognition},
	elocation-id = {2024.04.02.587669},
	year = {2024},
	doi = {10.1101/2024.04.02.587669},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/04/10/2024.04.02.587669},
	eprint = {https://www.biorxiv.org/content/early/2024/04/10/2024.04.02.587669.full.pdf},
	journal = {bioRxiv}
}"""


class _Maniquet2024ConfusionSimilarity(BenchmarkBase):
    """
    A benchmark class to measure the similarity between model-generated confusion probabilities
    and human confusion data in visual tasks, specifically designed for the Maniquet2024 dataset.

    Attributes:
        _metric (ConfusionSimilarity): The metric used to compare model outputs with human data.
        _fitting_stimuli (StimulusSet): Stimulus set used for training or fitting the model.
        _stimulus_set (StimulusSet): Stimulus set used for testing the model.
        _human_assembly (DataAssembly): Human behavioral data for comparison.
        _visual_degrees (int): The size of stimuli in visual degrees as presented to humans.
        _number_of_trials (int): Number of trials to average over for the model predictions.
    """

    def __init__(self):
        """
        Initializes the benchmark by setting up the necessary parameters.
        """
        # Initialize the metric for evaluating confusion similarity
        self._metric = load_metric('confusion_similarity')

        # Load training stimuli from the stimulus set registry
        self._fitting_stimuli = LazyLoad(lambda: load_stimulus_set('Maniquet2024-train'))

        # Load testing stimuli from the stimulus set registry
        self._stimulus_set = LazyLoad(lambda: load_stimulus_set('Maniquet2024-test'))

        # Load human behavioral data from the data registry
        self._human_assembly = LazyLoad(lambda: load_dataset('Maniquet2024'))

        # Set the visual degrees to which the human data was exposed
        self._visual_degrees = 8

        # Set the number of trials to perform
        self._number_of_trials = 1

        # Call the parent class constructor to complete initialization
        super(_Maniquet2024ConfusionSimilarity, self).__init__(
            identifier="_Maniquet2024ConfusionSimilarity",
            version=1,
            ceiling_func=lambda: self._metric._ceiling(self._assembly),
            parent="Maniquet2024",
            bibtex=BIBTEX,
        )

    def __call__(self, candidate: BrainModel):
        """
        Executes the benchmark by comparing the candidate model's confusion probabilities against human data.

        Args:
            candidate (BrainModel): The model being evaluated.

        Returns:
            float: The similarity score between the model and human data.
        """
        # Start the model on the task of predicting confusion probabilities
        candidate.start_task(BrainModel.Task.probabilities, self._fitting_stimuli)

        # Prepare the stimulus set by placing it on a virtual screen at a scale appropriate for the model
        stimulus_set = place_on_screen(
            self._stimulus_set,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees,
        )

        # Model looks at the stimulus set and returns confusion probabilities
        probabilities = candidate.look_at(
            stimulus_set, number_of_trials=self._number_of_trials
        )

        # Compute the confusion similarity score between model probabilities and human assembly data
        score = self._metric(probabilities, self._human_assembly)

        return score


class Maniquet2024TasksConsistency(BenchmarkBase):
    """
    A benchmarking class designed to evaluate the consistency of the human accuracy profiles across
    all tasks with the model's accuracy profiles across the same tasks.

    Attributes:
        _metric (TasksConsistency): The metric for evaluating task consistency between the model and human data.
        _fitting_stimuli (StimulusSet): The set of stimuli used for model training or calibration.
        _stimulus_set (StimulusSet): The set of stimuli used for testing the model's predictions.
        _human_assembly (DataAssembly): The dataset containing human response data for comparison.
        _visual_degrees (int): The visual size of the stimuli as perceived by human subjects.
        _number_of_trials (int): The number of trials over which model predictions are averaged.
    """

    def __init__(self):
        """
        Initializes the benchmark setup, including loading necessary datasets, defining the metric, and setting
        up the parameters for the evaluation.
        """
        # Metric for evaluating the consistency of task performance
        self._metric = load_metric('tasks_consistency')

        # Load training stimuli from the stimulus set registry
        self._fitting_stimuli = LazyLoad(lambda: load_stimulus_set('Maniquet2024-train'))

        # Load testing stimuli from the stimulus set registry
        self._stimulus_set = LazyLoad(lambda: load_stimulus_set('Maniquet2024-test'))

        # Load human behavioral data from the data registry
        self._human_assembly = LazyLoad(lambda: load_dataset('Maniquet2024'))

        # Set the visual context to match human study conditions
        self._visual_degrees = 8

        # Define the number of trials for model evaluation
        self._number_of_trials = 1

        # Initialize parent class with benchmark-specific metadata
        super(Maniquet2024TasksConsistency, self).__init__(
            identifier="Maniquet2024TasksConsistency",
            version=1,
            ceiling_func=lambda: self._metric.ceiling(self._human_assembly),
            parent="Maniquet2024",
            bibtex=BIBTEX,
        )

    def __call__(self, candidate: BrainModel):
        """
        Executes the benchmark by comparing the candidate model's task performance probabilities
        against human data, and returns a similarity score.

        Args:
            candidate (BrainModel): The neural model being evaluated.

        Returns:
            float: A similarity score indicating how closely the model's responses match human responses.
        """
        # Task the model with generating predictions based on the fitting stimuli
        candidate.start_task(BrainModel.Task.probabilities, self._fitting_stimuli)

        # Adjust the stimulus presentation to match the model's expected input scale
        stimulus_set = place_on_screen(
            self._stimulus_set,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees,
        )

        # Obtain the model's predictions as confusion probabilities
        probabilities = candidate.look_at(
            stimulus_set, number_of_trials=self._number_of_trials
        )

        # Evaluate the consistency of model predictions with human data
        score = self._metric(probabilities, self._human_assembly)

        return score


def Maniquet2024ConfusionSimilarity():
    return _Maniquet2024ConfusionSimilarity
