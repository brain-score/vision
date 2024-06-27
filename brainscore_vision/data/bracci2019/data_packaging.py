#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:51:41 2024

@author: costantino_ai
"""

# Imports necessary libraries
import os
import logging
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet
from brainio.packaging import package_data_assembly, package_stimulus_set
from natsort import natsorted

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
ROOT_DIRECTORY = "./bracci2019/private"
TAG = "Bracci2019"


# Function to categorize stimulus based on its ID
def categorize_stimulus(stim_id):
    if stim_id < 10:
        return "lookalike"
    elif 10 <= stim_id <= 18:
        return "animal"
    elif 19 <= stim_id <= 27:
        return "object"
    else:
        raise ValueError("Stimulus ID is out of the expected range")


def categorize_group(stim_id):
    if stim_id % 9 == 0:
        return 9
    return stim_id % 9


def load_stimulus_set(stimuli_directory, tag):
    """
    Scans a specified directory for JPEG images and organizes their metadata into a structured format.

    This function navigates to a directory named 'stimuli' within a given root directory,
    extracts metadata from the filenames, and categorizes each image based on predefined
    criteria. The metadata includes stimulus IDs, categories, and group affiliations, and
    is returned as a StimulusSet object.

    :param root: The root directory where the 'stimuli' directory is located.
    :type root: str
    :return: A StimulusSet object containing metadata and mappings for each stimulus.
    :rtype: StimulusSet
    """

    # Initialize lists to collect metadata and mapping of stimulus IDs to file paths
    stimuli = []
    stimulus_paths = {}

    # Iterate over each image file in the stimuli directory
    for filepath in natsorted(Path(stimuli_directory).glob("*.jpg")):
        stimulus_id = filepath.stem
        _, stim_id_str = stimulus_id.split("_")
        stim_id = int(stim_id_str)

        # Maps each stimulus ID to its file path
        stimulus_paths[stimulus_id] = filepath

        # Collects metadata for each stimulus
        stimuli.append(
            {
                "stimulus_id": stimulus_id,
                "stimulus_name": f"{stim_id:02d}_{categorize_group(stim_id):02d}_{categorize_stimulus(stim_id)}",
                "exemplar_number": stim_id,
                "image_label": categorize_stimulus(stim_id),
                "image_group": categorize_group(stim_id),
            }
        )

    # Convert the list of stimuli into a StimulusSet object
    stimulus_set = StimulusSet(stimuli)
    stimulus_set.stimulus_paths = stimulus_paths
    stimulus_set.name = tag
    assert len(stimulus_set) == 27
    logging.info("Total stimuli loaded: %d", len(stimulus_set))

    return stimulus_set


def load_brain_data(mat_file_path, stimulus_set, assembly_name):
    """
    Load brain imaging data from a MATLAB file and organize it into a structured data assembly.

    This function retrieves brain activity data segmented by regions of interest (ROIs) from a specified MATLAB file.
    It concatenates data across three predefined ROIs (v1, postVTC, antVTC) for each subject and maps it to the
    corresponding stimulus metadata. The resulting NeuroidAssembly object is structured to facilitate further analysis.

    Parameters:
    - mat_file_path (str): Full path to the MATLAB '.mat' file containing the brain data.
    - stimulus_set (pd.DataFrame): DataFrame containing stimulus metadata.
    - assembly_name (str): Name to assign to the resulting NeuroidAssembly for identification.

    Returns:
    - NeuroidAssembly: An object containing the organized brain data, along with metadata about stimuli, subjects, and ROIs.

    Raises:
    - FileNotFoundError: If the specified '.mat' file does not exist.
    - AssertionError: To ensure the stimulus IDs in the data assembly match those in the provided stimulus set.
    """

    # Load the MATLAB file
    mat_file = loadmat(mat_file_path)

    # Extract brain data for each ROI
    v1_data = mat_file["lookalike"]["data"][0][0][0][0][0]
    postVTC_data = mat_file["lookalike"]["data"][0][0][0][0][1]
    antVTC_data = mat_file["lookalike"]["data"][0][0][0][0][2]

    # Log the shape of data from each ROI for debugging
    print(
        f"V1 data shape: {v1_data.shape}, PostVTC data shape: {postVTC_data.shape}, AntVTC data shape: {antVTC_data.shape}"
    )

    # Concatenate data across the voxel dimension for each subject
    concatenated_data = np.concatenate([v1_data, postVTC_data, antVTC_data], axis=1)

    # Calculate number of conditions and subjects from one of the ROIs
    n_conditions, total_voxels, n_subjects = concatenated_data.shape

    # Initialize arrays to store indices
    subject_indices = []
    roi_indices = []
    voxel_indices = []

    # Generate indices arrays for ROI, subjects, and voxels
    for subject in range(n_subjects):
        total_voxels_per_subject = 0
        for roi, roi_data in enumerate([v1_data, postVTC_data, antVTC_data]):
            n_voxels = roi_data.shape[1]
            roi_indices.extend([roi] * n_voxels)
            subject_indices.extend([subject] * n_voxels)
            total_voxels_per_subject += n_voxels

        voxel_indices.extend(list(range(total_voxels_per_subject)))

    # Flatten the concatenated data into two dimensions
    flattened_data = concatenated_data.reshape(n_conditions, -1)

    # Convert index lists to numpy arrays for efficient data handling
    subject_indices = np.array(subject_indices)
    roi_indices = np.array(roi_indices)

    # Create the NeuroidAssembly object with corresponding dimensions and coordinates
    assembly = NeuroidAssembly(
        flattened_data,
        dims=["presentation", "neuroid"],
        coords={
            "stimulus_id": ("presentation", stimulus_set["stimulus_id"].values),
            "stimulus_name": ("presentation", stimulus_set["stimulus_name"].values),
            "exemplar_number": ("presentation", stimulus_set["exemplar_number"].values),
            "image_label": ("presentation", stimulus_set["image_label"].values),
            "image_group": ("presentation", stimulus_set["image_group"].values),
            "roi": ("neuroid", roi_indices),
            "subject": ("neuroid", subject_indices),
            "voxels": ("neuroid", voxel_indices),
        },
    )

    # Assign a name to the data assembly
    assembly.name = assembly_name

    # Ensure the assembly's stimulus IDs match those provided in the stimulus set
    assert np.array_equal(
        assembly["stimulus_id"].values, stimulus_set["stimulus_id"].values
    ), "Stimulus IDs do not match."

    return assembly


def main():
    """
    Main function to package stimulus set and experimental data, and upload to S3.
    """
    logging.info("Starting the data packaging process.")

    # Load stimuli from directory
    human_stimuli_directory = os.path.join(ROOT_DIRECTORY, "stimuli")
    human_stimulus_set = load_stimulus_set(human_stimuli_directory, TAG)

    # Upload stimuli
    human_stimulus_meta = package_stimulus_set(
        None, human_stimulus_set, human_stimulus_set.name, "brainio-brainscore"
    )

    # Load human data assembly
    brain_data_path = os.path.join(ROOT_DIRECTORY, "data/lookalike.mat")
    data_assembly = load_brain_data(brain_data_path, human_stimulus_set, TAG)

    assembly_meta = package_data_assembly(
        None,
        data_assembly,
        data_assembly.name,
        human_stimulus_set.name,
        "NeuroidAssembly",
        "brainio-brainscore",
    )

    print(human_stimulus_meta)
    print(assembly_meta)


if __name__ == "__main__":
    main()
