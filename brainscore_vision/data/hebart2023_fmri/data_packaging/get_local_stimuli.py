##### HEBART2023_FMRI LOCAL STIMULI PACKAGING #####
# In order to run the Hebart2023_fmri benchmark locally, you need to obtain the stimuli yourself due to licensing rules.
# Only the neural data can be downloaded from the public S3 bucket.
# This file packages the stimuli into Brain-Score format and saves them at the BrainIO cache location.

### STEPS FOR RUNNING HEBART2023_FMRI LOCALLY:
# 1) Download the THINGS image database from https://osf.io/jum2f/
# -> You need images_THINGS.zip (ca. 4.7 GB) and password_images.txt.
# -> Confirm that you have read the LICENSE and unzip the images with the password provided in password_images.txt.
# -> Your file structure should now look something like THINGS_IMAGE_ROOT/object_images/... 
#    (e.g. 'THINGS_IMAGE_ROOT/object_images/aardvark/aardvark_01b.jpg')
# 2) You also need the stimulus metadata csv file for at least one subject, we provide one in this directory 
#    (source: https://plus.figshare.com/articles/dataset/THINGS-data_fMRI_Single_Trial_Responses_table_format_/20492835?file=43635873)
# 3) Set the paths correctly.
# 4) Run this script.
# 5) Update sha1 in the __init__.py if necessary
#    ATTENTION: sha1 hashes of the csv files will probably change
#    The script will adapt to this by renaming the cache dirs to the new sha1, 
#    but you will need to change the parameter "csv_sha1" of the associated stimulus_set_registry entry in __init__.py
# 6) Run your benchmark.

### OUTPUT
# Stimulus sets for train and test splits in Brain-Score format.
# In the end, four directories must exist in your BrainIO cache location.
# load_stimulus_set_from_s3 looks for directories based on the SHA1 hashes.
# The default location is BRAINIO_HOME = ~/.brainio/

### CONFIRM YOUR CACHE LOCATION LOOKS LIKE THIS:
# train stimuli csv: <csv_sha1>/stimulus_THINGS_fMRI_train_Stimuli.csv
# train stimuli zip: <zip_sha1>/stimulus_THINGS_fMRI_train_Stimuli.zip and stimulus_THINGS_fMRI_train_Stimuli directory
# test stimuli csv: <csv_sha1>/stimulus_THINGS_fMRI_test_Stimuli.csv
# test stimuli zip: <zip_sha1>/stimulus_THINGS_fMRI_test_Stimuli.zip and stimulus_THINGS_fMRI_test_Stimuli directory

### INPUTS:
# - things_image_dir: Directory containing THINGS images.
# - metadata_dir: Directory containing subject metadata CSV files. (default: this directory)
# - output_path: Directory to save the BrainIO cache. (default: ~/.brainio/)

# ORIGINAL SHA1 INFORMATION
TRAIN_CSV_SHA1 = "b424b1a55595a4666fbc140a5a801fcd184d1a44"
TRAIN_ZIP_SHA1 = "1c65c28c104e100e6e3fe2128656abe647e41bd9"
TEST_CSV_SHA1 = "0ef71c62210d0a0bf91cb2cd8f0e1404477e0e3a"
TEST_ZIP_SHA1 = "41cd5a05fe9d5118b05fc6b35654e36ff021570d"

import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from brainscore_core.supported_data_standards.brainio.packaging import create_stimulus_csv, create_stimulus_zip
from brainscore_core.supported_data_standards.brainio.fetch import unzip
from package_hebart2023fmri import get_stimulus_set


SPLITS = ['train', 'test']


def process_subject_data(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the train and test stimuli from the metadata
    Adapted from process_subject_data in package_gifford2022.py
    """

    # Get sorting indices w.r.t. stimulus filenames
    # and split the data into train and test sets
    indices = metadata['stimulus'].argsort()
    train_mask = ((metadata['trial_type'] == 'train').values)[indices]
    train_indices = indices[train_mask]
    test_indices = indices[~train_mask]
    
    # Get the train and test stimuli
    train_stimuli = metadata['stimulus'][train_indices].values
    test_stimuli = metadata['stimulus'][test_indices].values
    test_stimuli = test_stimuli.reshape(-1, 12)
    
    # Verify test stimuli consistency across repetitions
    assert np.all(test_stimuli == test_stimuli[:, [0]]), "Test stimuli not consistent across repetitions"
    test_stimuli = test_stimuli[:, 0]    

    return train_stimuli, test_stimuli


def create_local_cache(metadata_dir: str, things_image_dir: str, output_path: str) -> None:
    """
    Create local stimulus cache for the Hebart2023_fmri benchmark.

    Args:
        metadata_dir: Path to the directory containing subject metadata CSV files.
        things_image_dir: Path to the THINGS image database directory.
        output_path: Path to the output BrainIO cache directory.
    """
    metadata_dir = Path(metadata_dir).expanduser()
    things_image_dir = Path(things_image_dir).expanduser() / "object_images"
    output_path = Path(output_path).expanduser()

    # metadata is identical across subjects, we provide sub-01 
    metadata_file = metadata_dir / f"sub-01_StimulusMetadata.csv"
    metadata = pd.read_csv(metadata_file)
    train_stimuli, test_stimuli = process_subject_data(metadata)

    for split in SPLITS:
        if split == 'train':
            stimuli = train_stimuli
            csv_sha1 = TRAIN_CSV_SHA1
            zip_sha1 = TRAIN_ZIP_SHA1
        elif split == 'test':
            stimuli = test_stimuli
            csv_sha1 = TEST_CSV_SHA1
            zip_sha1 = TEST_ZIP_SHA1
        else:
            raise ValueError(f"Unknown split: {split}")

        stimulus_set = get_stimulus_set(
            things_image_dir=things_image_dir,
            image_paths=stimuli,
            split=split)
        
        #### ADAPTED FROM brainscore_core/supported_data_standards/brainio/packaging.py package_stimulus_set
        
        # naming
        stimulus_set_identifier= stimulus_set.name
        stimulus_store_identifier = "stimulus_" + stimulus_set_identifier.replace(".", "_")
        # - csv
        csv_file_name = stimulus_store_identifier + ".csv"
        target_csv_path = output_path / csv_sha1 / csv_file_name
        # - zip
        zip_file_name = stimulus_store_identifier + ".zip"
        target_zip_path = output_path / zip_sha1 / zip_file_name
        
        os.makedirs(target_csv_path.parent, exist_ok=True)
        os.makedirs(target_zip_path.parent, exist_ok=True)

        # create csv and zip files
        stimulus_zip_sha1, zip_filenames = create_stimulus_zip(stimulus_set, str(target_zip_path))
        print(f"Created stimulus zip at {target_zip_path} with sha1: {stimulus_zip_sha1}")

        unzip(target_zip_path)

        assert 'filename' not in stimulus_set.columns, "StimulusSet already has column 'filename'"
        stimulus_set['filename'] = zip_filenames  # keep record of zip (or later local) filenames
        stimulus_csv_sha1 = create_stimulus_csv(stimulus_set, str(target_csv_path))
        
        # rename the dir to stimulus_csv_sha1
        if csv_sha1 != stimulus_csv_sha1:
            print("CSV sha1 changed, renaming directory...")
            new_dir = output_path / stimulus_csv_sha1
            os.rename(target_csv_path.parent, new_dir)
            target_csv_path = new_dir / csv_file_name
            print(f"Please update the __init__.py file!")
        print(f"Created stimulus csv at {target_csv_path} with sha1: {stimulus_csv_sha1}")
        
if __name__ == "__main__":
    parser = ArgumentParser(description="Create local stimulus cache for Hebart2023_fmri benchmark")
    parser.add_argument(
        '--metadata-dir', '--meta-dir',
        dest='metadata_dir',
        type=str,
        default='.',
        help='Path to the neural data dir with the metadata file.'
    )
    parser.add_argument(
        '--things-image-dir', '--imgs-dir',
        dest='things_image_dir',
        type=str,
        required=True,
        help='Path to the THINGS image database directory containing images.'
    )
    parser.add_argument(
        '--output-dir', '--out-dir',
        dest='output_dir',
        type=str,
        default='~/.brainio/',
        help='Path to the output brainio cache directory.'
    )
    args = parser.parse_args()

    create_local_cache(
        metadata_dir=args.metadata_dir,
        things_image_dir=args.things_image_dir,
        output_path=args.output_dir
    )