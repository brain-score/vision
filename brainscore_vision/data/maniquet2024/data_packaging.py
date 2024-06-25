#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 00:24:30 2024

@author: costantino_ai
"""

import os
import logging
from pathlib import Path
import pandas as pd
from brainio.assemblies import BehavioralAssembly
from brainio.stimuli import StimulusSet
from brainio.packaging import package_data_assembly, package_stimulus_set

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
ROOT_DIRECTORY = "./maniquet2024/private"
TAG = "Maniquet2024"


def load_stimulus_set(stimuli_directory, tag):
    """
    Load and package stimuli from the specified directory.

    Args:
        stimuli_directory (str): Directory containing stimulus files.
        tag (str): Tag to assign to the stimulus set.

    Returns:
        StimulusSet: Packaged set of stimuli with metadata.
    """
    logging.info("Loading stimuli from directory: %s", stimuli_directory)
    stimuli = []
    stimulus_paths = {}

    for filepath in Path(stimuli_directory).glob("*.png"):
        stimulus_id = filepath.stem
        parts = filepath.stem.split("_")
        exemplar_number, manipulation, manipulation_details, category = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
        )

        stimulus_paths[stimulus_id] = filepath
        stimuli.append(
            {
                "stimulus_id": stimulus_id,
                "manipulation": manipulation,
                "manipulation_details": manipulation_details,
                "image_label": category,
                "exemplar_number": exemplar_number,
            }
        )

    stimulus_set = StimulusSet(stimuli)
    stimulus_set.stimulus_paths = stimulus_paths
    stimulus_set.name = tag
    logging.info("Total stimuli loaded: %d", len(stimulus_set))
    return stimulus_set


def load_behavioral_data(data_file, tag):
    """
    Load and package experimental data from a CSV file.

    Args:
        data_file (str): Path to the CSV file containing experimental data.
        tag (str): Tag to assign to the behavioral data assembly.

    Returns:
        BehavioralAssembly: Data assembly of behavioral responses.
    """
    logging.info("Loading behavioral data from file: %s", data_file)
    df = pd.read_csv(data_file)
    assembly = BehavioralAssembly(
        df["acc"],
        dims=["presentation"],
        coords={
            "stimulus_id": ("presentation", df["stimulus_id"].values),
            "manipulation": ("presentation", df["condition"].values),
            "manipulation_details": ("presentation", df["task_details"].values),
            "mask": ("presentation", df["mask"].values),
            "image_label": ("presentation", df["category"].values),
            "prediction": ("presentation", df["prediction"].values),
            "response": ("presentation", df["response"].values),
            "reaction_time": ("presentation", df["rt"].values),
            "subject_id": ("presentation", df["subj"].values),
            "task": ("presentation", df["task_long"].values),
        },
    )
    assembly.name = tag
    logging.info(
        "Data assembly loaded with %d presentations", len(assembly["presentation"])
    )
    return assembly


def main():
    """
    Main function to package stimulus set and experimental data, and upload to S3.
    """
    logging.info("Starting the data packaging process.")

    # Load stimuli from directories
    human_stimuli_directory = os.path.join(ROOT_DIRECTORY, "human_stimuli")
    dnntest_stimuli_directory = os.path.join(ROOT_DIRECTORY, "dnn_stimuli/test")
    dnntrain_stimuli_directory = os.path.join(ROOT_DIRECTORY, "dnn_stimuli/train")

    human_stimulus_set = load_stimulus_set(human_stimuli_directory, TAG)
    dnntest_stimulus_set = load_stimulus_set(dnntest_stimuli_directory, f"{TAG}-test")
    dnntrain_stimulus_set = load_stimulus_set(dnntrain_stimuli_directory, f"{TAG}-train")

    # Upload stimuli
    human_stimulus_meta = package_stimulus_set(
        None, human_stimulus_set, human_stimulus_set.name, "brainio-brainscore"
    )
    dnntest_stimulus_meta = package_stimulus_set(
        None, dnntest_stimulus_set, dnntest_stimulus_set.name, "brainio-brainscore"
    )
    dnntrain_stimulus_meta = package_stimulus_set(
        None, dnntrain_stimulus_set, dnntrain_stimulus_set.name, "brainio-brainscore"
    )

    # Load human data assembly
    data_file = os.path.join(ROOT_DIRECTORY, "data/human_data_andrea.csv")
    data_assembly = load_behavioral_data(data_file, TAG)
    assembly_meta = package_data_assembly(
        None,
        data_assembly,
        data_assembly.name,
        human_stimulus_set.name,
        "BehavioralAssembly",
        "brainio-brainscore",
    )

    # print(human_stimulus_meta)
    # print(dnntest_stimulus_meta)
    # print(dnntrain_stimulus_meta)
    # print(assembly_meta)


if __name__ == "__main__":
    main()
