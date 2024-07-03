#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 22:18:07 2024

@author: costantino_ai
"""

import xarray as xr
import numpy as np
from scipy.stats import spearmanr
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.model_interface import BrainModel
from brainscore_vision import load_stimulus_set, load_metric, load_dataset
from brainscore_vision.utils import LazyLoad
from brainio.assemblies import NeuroidAssembly
from brainscore_core.metrics import Score


BIBTEX = """@article{bracci2019ventral,
  title={The ventral visual pathway represents animal appearance over animacy, unlike human behavior and deep neural networks},
  author={Bracci, Stefania and Ritchie, J Brendan and Kalfas, Ioannis and de Beeck, Hans P Op},
  journal={Journal of Neuroscience},
  volume={39},
  number={33},
  pages={6513--6525},
  year={2019},
  publisher={Soc Neuroscience}
}"""

TIME_BIN_ST, TIME_BIN_END = (
    70,
    170,
)  # standard core object recognition response, following Majaj*, Hong*, et al. 2015


class _Bracci2019RSA(BenchmarkBase):
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

    def __init__(self, region):
        """
        Initializes the benchmark by setting up the necessary parameters.
        """

        # Initialize the metric for evaluating confusion similarity
        self._metric = load_metric("rdm")

        # # Load testing stimuli from the stimulus set registry
        # self._stimulus_set = stimulus_set_registry["Bracci2019"]()

        # # Load human behavioral data from the data registry
        # self._human_assembly = data_registry["Bracci2019"]()

        # Load testing stimuli from the stimulus set registry
        self._stimulus_set = LazyLoad(lambda: load_stimulus_set('Bracci2019'))

        # Load human behavioral data from the data registry
        self._human_assembly = LazyLoad(lambda: load_dataset('Bracci2019'))

        # Set the visual degrees to which the human data was exposed
        self._visual_degrees = 8

        # Set the number of trials to perform
        self._number_of_trials = 1

        # Set the region to record from
        self._region = region

        # Define a mapping
        self._roi_map = {
            "V1": (0, "V1"),
            "posteriorVTC": (1, "IT"),
            "anteriorVTC": (2, "IT"),
        }

        assert (
            self._region in self._roi_map
        ), "The ROI to compare must either ['V1', 'posteriorVTC', or 'anteriorVTC']"

        # Call the parent class constructor to complete initialization
        super(_Bracci2019RSA, self).__init__(
            identifier=f"Bracci2019.{region}-rdm",
            version=1,
            # ceiling_func=lambda: self._metric._ceiling(self._assembly),
            ceiling_func=lambda: 1,
            parent="Bracci2019",
            bibtex=BIBTEX,
        )

    def _center_data_by_subject(self, roi_assembly):
        """
        Center the data by subject by subtracting the mean across all conditions for each voxel.

        Args:
            roi_assembly (DataArray): The input data array with regions of interest and subject identifiers.

        Returns:
            DataArray: The ROI data array with centered data by subject.
        """
        # Get unique list of subjects in the data
        subjects = np.unique(roi_assembly["subject"])
        centered_data_list = []

        for subject in subjects:
            # Select data for the current subject
            subject_data = roi_assembly.sel(neuroid=roi_assembly["subject"] == subject)
            # Center the data by subtracting the mean across presentations
            subject_centered_data = subject_data - subject_data.mean(dim="presentation")
            # Append the centered data for this subject to the list
            centered_data_list.append(subject_centered_data)

        # Concatenate all the subject-centered data back into a single assembly
        full_centered_assembly = xr.concat(centered_data_list, dim="neuroid")

        return full_centered_assembly

    def _get_human_ceiling(self, roi_centered_assembly):
        """
        Calculate the "lower bound" ceiling for human performance based on inter-subject correlations of RDMs.

        Args:
            roi_centered_assembly (DataArray): The ROI data array with subject-centered data.

        Returns:
            float: The average inter-subject correlation, representing the human performance ceiling.
        """
        # Get unique list of subjects in the data
        subjects = np.unique(roi_centered_assembly["subject"])
        correlations = []

        for subject in subjects:
            # Generate RDM for the current subject
            subject_rdm = self._metric._rdm(
                roi_centered_assembly.sel(
                    neuroid=roi_centered_assembly["subject"] == subject
                )
            ).values

            # Generate RDMs for other subjects
            other_rdms = [
                self._metric._rdm(
                    roi_centered_assembly.sel(
                        neuroid=roi_centered_assembly["subject"] == other_subject
                    )
                ).values
                for other_subject in subjects
                if other_subject != subject
            ]

            # Compute the average RDM from other subjects
            average_rdm = np.mean(other_rdms, axis=0)

            # Extract the lower triangle of the RDM matrix
            mask = np.tril(np.ones_like(subject_rdm), -1).astype(bool)
            vector_lt_subject = subject_rdm[mask]
            vector_lt_average = average_rdm[mask]

            # Compute the Spearman correlation
            correlation, _ = spearmanr(vector_lt_subject, vector_lt_average)

            # Append the correlation result
            correlations.append(correlation)

        # Compute the average correlation across all subjects
        human_ceiling = np.mean(correlations)

        return float(human_ceiling)

    def _average_voxels_across_subjects(self, roi_centered_assembly):
        """
        Compute the mean across the 'voxels' dimension to create a new NeuroidAssembly with averaged neuroid data.

        This function averages voxel data within the input neuroid assembly across all subjects and retains important metadata
        about presentations and neuroids.

        Args:
            roi_centered_assembly (xarray.DataArray): The input data array with regions of interest, voxels, and subject data.

        Returns:
            NeuroidAssembly: A new assembly object with averaged neuroid data and associated coordinates.
        """
        # Average voxel data across subjects to simplify the neuroid representation
        averaged_assy = roi_centered_assembly.groupby("voxels").mean()
        averaged_data = averaged_assy.values

        # Construct the NeuroidAssembly with the averaged data and maintain the presentation metadata
        assembly = NeuroidAssembly(
            averaged_data,
            dims=["presentation", "neuroid"],
            coords={
                "stimulus_id": (
                    "presentation",
                    roi_centered_assembly["stimulus_id"].values,
                ),
                "stimulus_name": (
                    "presentation",
                    roi_centered_assembly["stimulus_name"].values,
                ),
                "exemplar_number": (
                    "presentation",
                    roi_centered_assembly["exemplar_number"].values,
                ),
                "image_label": (
                    "presentation",
                    roi_centered_assembly["image_label"].values,
                ),
                "image_group": (
                    "presentation",
                    roi_centered_assembly["image_group"].values,
                ),
                "voxels": ("neuroid", list(range(averaged_data.shape[-1]))),
            },
        )
        return assembly

    def __call__(self, candidate: BrainModel):
        """

        Args:
            candidate (BrainModel): The model being evaluated.

        Returns:
            float: The similarity score between the model and human data.
        """
        # Start the model on the task of predicting confusion probabilities
        candidate.start_recording(
            self._roi_map[self._region][1], [(TIME_BIN_ST, TIME_BIN_END)]
        )

        # Prepare the stimulus set by placing it on a virtual screen at a scale appropriate for the model
        stimulus_set = place_on_screen(
            self._stimulus_set,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees,
        )

        # Model looks at the stimulus set
        dnn_assembly = candidate.look_at(
            stimulus_set, number_of_trials=self._number_of_trials
        )

        # Get the human data
        human_data = self._human_assembly

        # Select only the data for the current ROI
        roi_idx = self._roi_map[self._region][0]
        roi_assembly = human_data.sel(neuroid=human_data["roi"] == roi_idx)

        # Center data (by subject, across all conditions)
        roi_centered_assembly = self._center_data_by_subject(roi_assembly)

        # Average voxels across subjects
        human_averaged_assembly = self._average_voxels_across_subjects(
            roi_centered_assembly
        )

        # Calculate the human ceiling
        ceiling = self._get_human_ceiling(roi_centered_assembly)

        # Compare (corr) the two RDMs to get the score
        similarity = self._metric(dnn_assembly, human_averaged_assembly)

        # Normalize by ceiling
        score = Score(similarity / ceiling)
        score.attrs["raw"] = similarity
        score.attrs["ceiling"] = ceiling

        return score


def Bracci2019RSA(region):
    return _Bracci2019RSA(region)
