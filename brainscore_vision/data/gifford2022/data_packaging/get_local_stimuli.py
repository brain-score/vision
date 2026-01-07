##### GIFFORD2022 LOCAL STIMULI PACKAGING #####
# In order to run the Gifford2022 benchmark locally, you need to obtain the stimuli yourself due to licensing rules.
# Only the neural data can be downloaded from the public S3 bucket.
# This file packages the stimuli into Brain-Score format and saves them at the BrainIO cache location.

### STEPS FOR RUNNING GIFFORD2022 LOCALLY:
# 1) Download the THINGS image database from https://osf.io/jum2f/
# -> You need images_THINGS.zip (ca. 4.7 GB) and password_images.txt.
# -> Confirm that you have read the LICENSE and unzip the images with the password provided in password_images.txt.
# -> Your file structure should now look something like THINGS_IMAGE_ROOT/object_images/... 
#    (e.g. 'THINGS_IMAGE_ROOT/object_images/aardvark/aardvark_01b.jpg')
# 2) If you want to replicate the exact experimental conditions, resize all images to 500x500 pixels
# -> use the create_THINGS_resized.py script in this directory
# -> provide the root of the new database to this script.
# 3) You also need the image metadata file image_metadata.npy provided in this dir (source: https://osf.io/y63gw/files ).
# 4) Set the paths correctly.
# 5) Run this script.
# 6) Update sha1 in the __init__.py if necessary
#    ATTENTION: sha1 hashes of the csv files will probably change
#    The script will adapt to this by renaming the cache dirs to the new sha1, 
#    but you will need to change the parameter "csv_sha1" of the associated stimulus_set_registry entry in __init__.py
# 7) Run your benchmark.

### OUTPUT
# Stimulus sets for train and test splits in Brain-Score format.
# In the end, four directories must exist in your BrainIO cache location.
# load_stimulus_set_from_s3 looks for directories based on the SHA1 hashes.
# The default location is BRAINIO_HOME = ~/.brainio/

### CONFIRM YOUR CACHE LOCATION LOOKS LIKE THIS:
# train stimuli csv: <csv_sha1>/stimulus_THINGS_EEG2_train_Stimuli.csv
# train stimuli zip: <zip_sha1>/stimulus_THINGS_EEG2_train_Stimuli.zip and stimulus_THINGS_EEG2_train_Stimuli directory
# test stimuli csv: <csv_sha1>/stimulus_THINGS_EEG2_test_Stimuli.csv
# test stimuli zip: <zip_sha1>/stimulus_THINGS_EEG2_test_Stimuli.zip and stimulus_THINGS_EEG2_test_Stimuli directory

### INPUTS:
# - things_image_dir: Directory containing THINGS images.
# - metadata_dir: Directory containing the image_metadata.npy file.
# - output_path: Directory to the output BrainIO cache directory.


# ORIGINAL SHA1 INFORMATION
TRAIN_CSV_SHA1 = "1a10e75ef9dc9eed6a4eca8e183b81f0d642dda8"
TRAIN_ZIP_SHA1 = "7dcc25442ab6737302f96a7b3bd2526afd331637"
TEST_CSV_SHA1 = "d7325a55602239e67892b8c596d37e2ee609a59a"
TEST_ZIP_SHA1 = "bcd0b909df89ff39836ad0bcc9581a79189af2ac"

import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from brainscore_core.supported_data_standards.brainio.packaging import create_stimulus_csv, create_stimulus_zip
from brainscore_core.supported_data_standards.brainio.fetch import unzip
from package_gifford2022 import get_stimulus_set

SPLITS = ['train', 'test']

def create_local_cache(metadata_dir: str, things_image_dir: str, output_path: str) -> None:
    """
    Create local stimulus cache for the Gifford2022 benchmark.

    Args:
        metadata_dir: Path to the directory containing image_metadata.npy.
        things_image_dir: Path to the THINGS image database directory.
        output_path: Path to the output BrainIO cache directory.
    """
    metadata_dir = Path(metadata_dir).expanduser()
    things_image_dir = Path(things_image_dir).expanduser() / "object_images"
    output_path = Path(output_path).expanduser()

    image_metadata_path = metadata_dir / "image_metadata.npy"
    image_metadata = np.load(image_metadata_path, allow_pickle=True, fix_imports=True).item()

    print(f"Number of training stimuli: {len(image_metadata['train_img_files'])}, Number of testing stimuli: {len(image_metadata['test_img_files'])}")
    print(f"Example stimulus file: {image_metadata['train_img_files'][0]}, concept: {image_metadata['train_img_concepts'][0]}")


    for split in SPLITS:
        stimulus_set = get_stimulus_set(
            image_metadata=image_metadata,
            things_image_dir=things_image_dir,
            split=split,
        )

        if split == 'train':
            csv_sha1 = TRAIN_CSV_SHA1
            zip_sha1 = TRAIN_ZIP_SHA1
        elif split == 'test':
            csv_sha1 = TEST_CSV_SHA1
            zip_sha1 = TEST_ZIP_SHA1
        else:
            raise ValueError(f"Unknown split: {split}")

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
    parser = ArgumentParser(description="Create local stimulus cache for Gifford2022 benchmark")
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