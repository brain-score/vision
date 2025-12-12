import os
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
from tqdm.auto import tqdm

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.packaging import package_data_assembly, package_stimulus_set

#####
# Package the Gifford et al. (2022) EEG2 dataset based on THINGS images
# neuroids: 10 subjects x 17 channels = 170 neuroids
# train stimuli: 66160 presentations (1654 categories x 10 images x 4 repetitions)
# test stimuli: 16000 presentations (200 images x 80 repetitions)
# time bins: 100 time points (NOT flattened into neuroids), from -200ms to 800ms
# 
# 
# ATTENTION: The stimuli where packaged at 500x500 pixel resolution as they were shown in the original experiment.
# Use the create_THINGS_resized.py script to create a copy of the THINGS database with the correct resolution.
# Set --things-image-dir to the path of the resized THINGS images.
#####


DEFAULT_BUCKET_NAME = "brainscore-storage/brainio-brainscore"
DEFAULT_OUTPUT_PATH = "./uploaded_data_info/"

SPLITS = ['train', 'test']
SUBJECTS = ["sub-{:02d}".format(i) for i in range(1, 11)]

def get_stimulus_set(image_metadata: dict, things_image_dir: Path, split: str) -> StimulusSet:
    """
    Build the StimulusSet for the given split. THINGS images must be downloaded and unzipped with the password
    
    Args:
        image_metadata: Metadata of the stimuli provided by the neural dataset
        things_image_dir: Path to the stimulus data directory containing images
        split: 'train' or 'test'
    
    Returns:
        StimulusSet for the specified split
    """
    
    # Check that the THINGS image dataset is structured correctly
    file_name = image_metadata[f'{split}_img_files'][0]
    stimulus_id = file_name.split('.')[0]
    img_class = stimulus_id.rsplit('_', 1)[0]  # Get the class from the stimulus ID
    check_path = things_image_dir / img_class / file_name
    assert check_path.exists(), f"No image at: {check_path}"

    # Get the stimuli to populate each StimulusSet
    file_names = image_metadata[f'{split}_img_files']
    
    stimulus_ids, stimulus_paths, labels = [], {}, {}
    for file_name in tqdm(file_names, total=len(file_names), desc=f"Loading {split} stimuli"):

        stimulus_id = file_name.split('.')[0]
        img_class = stimulus_id.rsplit('_', 1)[0]  # Get the class from the stimulus ID
        stimulus_path = things_image_dir / img_class / file_name
        stimulus_ids.append(stimulus_id)
        stimulus_paths[stimulus_id] = stimulus_path
        labels[stimulus_id] = img_class
    
    
    labels2idx = {label: idx for idx, label in enumerate(np.unique(list(labels.values())))}
    labels_idx = {stimulus_id: labels2idx[label] for stimulus_id, label in labels.items()}

    stimuli = []
    for stimulus_id in stimulus_ids:
        stimuli.append({
            'stimulus_id': stimulus_id,
            'label': labels[stimulus_id],
            'object_name': labels[stimulus_id],
            'label_idx': labels_idx[stimulus_id],
        })
        
    stimulus_set = StimulusSet(stimuli)
    stimulus_set.stimulus_paths = stimulus_paths
    stimulus_set.name = f'THINGS_EEG2_{split}_Stimuli'
    stimulus_set.identifier = f'things_eeg2_{split}_stimuli'

    return stimulus_set

def get_neuroid_assembly(neural_data_dir: Path, stimulus_set: StimulusSet, split: str) -> NeuroidAssembly:
    """
    Create the NeuroidAssembly for the given split with time bins as a separate dimension.
    
    Args:
        neural_data_dir: Path to the neural data directory
        stimulus_set: StimulusSet for the split
        split: 'train' or 'test'
    
    Returns:
        NeuroidAssembly for the specified split with shape (presentation, neuroid, time_bin)
    """
    
    print(f"Processing {split} split across all subjects...")
    if split == 'train':
        filename = "preprocessed_eeg_training.npy"
    elif split == 'test':
        filename = "preprocessed_eeg_test.npy"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    data_neural_concat = []
    subject_names, channel_names, neuroid_indices = [], [], []
    for subj in tqdm(SUBJECTS, desc=f"Processing {split} subjects"):
        neural_data = np.load(neural_data_dir / subj / filename, allow_pickle=True).item()

        subject_responses = neural_data['preprocessed_eeg_data']
        ch_names = neural_data['ch_names']
        times = neural_data['times']
        n_stimuli, n_trials, n_channels, n_time_points = subject_responses.shape

        subject_names.extend([subj] * n_channels)
        channel_names.extend(ch_names)
        neuroid_indices.extend(np.arange(n_channels))

        # Z-score normalization across trials and stimuli
        # subject_responses: shape (n_stimuli, n_trials, n_channels, n_time_points)
        mean = np.mean(subject_responses, axis=(0, 1))
        std = np.std(subject_responses, axis=(0, 1))
        subject_responses = (subject_responses - mean) / std 
        
        # Reshape to (presentations, channels, time_points) - keep time as separate dimension
        subject_responses = subject_responses.reshape(n_stimuli * n_trials, n_channels, n_time_points)
        if subj == 'sub-01': 
            print(f"Each subject's recordings have data shape {subject_responses.shape} after reshaping")
        data_neural_concat.append(subject_responses)

    # Concatenate across subjects
    data_neural_concat = np.concatenate(data_neural_concat, axis=1)
    trial_indices = np.tile(np.arange(n_trials), n_stimuli)
    print(f"Concatenated neural data shape for split {split}: {data_neural_concat.shape} "+\
          f"(#presentations = {n_stimuli} Stimuli x {n_trials} Trials, "+\
          f"#neuroids = {n_channels} Channels x {len(SUBJECTS)} Subjects, "+\
          f"#time_bins = {n_time_points} Timepoints)")

    total_neuroids = len(SUBJECTS) * n_channels  # 10 subjects * 17 channels
    total_presentations = n_stimuli * n_trials  # stimuli * trials
    assert np.all([
        len(subject_names) == len(channel_names) == len(neuroid_indices) == data_neural_concat.shape[1] == total_neuroids,
        len(trial_indices) == data_neural_concat.shape[0] == total_presentations,
        data_neural_concat.shape[2] == n_time_points
    ])

    time_bin_start = np.round(times, decimals=2) # prevent floating point precision problems
    time_bin_end = np.round(np.append(time_bin_start[1:], [0.8]), decimals=2)

    assembly = NeuroidAssembly(
        data_neural_concat,
        dims=["presentation", "neuroid", "time_bin"],
        coords={
            "stimulus_id": ("presentation", np.repeat(stimulus_set["stimulus_id"].values, n_trials)),
            "stimulus_label": ("presentation", np.repeat(stimulus_set["label"].values, n_trials)),
            "object_name": ("presentation", np.repeat(stimulus_set["object_name"].values, n_trials)),
            "stimulus_label_idx": ("presentation", np.repeat(stimulus_set["label_idx"].values, n_trials)),
            "repetition": ("presentation", trial_indices),
            "subject": ("neuroid", subject_names),
            "neuroid_id": ("neuroid", neuroid_indices),
            "channel": ("neuroid", channel_names),
            "time_bin_start": ("time_bin", time_bin_start),
            "time_bin_end": ("time_bin", time_bin_end)
        },
    )

    # Assign a name to the data assembly
    assembly.name = f'THINGS_EEG2_{split}_Assembly'
    assembly.attrs['stimulus_set'] = stimulus_set

    return assembly

def package_data(neural_data_dir: Path, things_image_dir: Path, bucket_name: str, output_path: Path) -> None:
    """
    Main function to package THINGS EEG2 data with time bins as separate dimension.
    
    Args:
        neural_data_dir: Path to the neural data directory containing preprocessed EEG .npy files
        things_image_dir: Path to the stimulus data directory containing images
        bucket_name: S3 bucket name for upload
        output_path: Directory path where file hashes will be saved
    """
    image_metadata_path = neural_data_dir / "image_metadata.npy"
    image_metadata = np.load(image_metadata_path, allow_pickle=True, fix_imports=True).item()

    print(f"Number of training stimuli: {len(image_metadata['train_img_files'])}, Number of testing stimuli: {len(image_metadata['test_img_files'])}")
    print(f"Example stimulus file: {image_metadata['train_img_files'][0]}, concept: {image_metadata['train_img_concepts'][0]}")

    # Print properties of the data being packaged, example subject sub-01
    training_subj_data = np.load(neural_data_dir / "sub-01" / "preprocessed_eeg_training.npy", allow_pickle=True).item()
    testing_subj_data = np.load(neural_data_dir / "sub-01" / "preprocessed_eeg_test.npy", allow_pickle=True).item()
    print(f"Time bins are shaped: {training_subj_data['times'].shape}, times: {training_subj_data['times'][:3]} ...")
    print(f"There are {len(training_subj_data['ch_names'])} channels: {training_subj_data['ch_names'][:3]} ...")
    print(f"Preprocessed train EEG data (4 repetitions of each stimulus) has shape {training_subj_data['preprocessed_eeg_data'].shape}")
    print(f"Preprocessed test EEG data (80 repetitions of each stimulus) has shape {testing_subj_data['preprocessed_eeg_data'].shape}")

    # Create stimulus sets and assemblies for both splits
    train_stimulus_set = get_stimulus_set(image_metadata, things_image_dir, split='train')
    test_stimulus_set = get_stimulus_set(image_metadata, things_image_dir, split='test')
    train_assembly = get_neuroid_assembly(neural_data_dir, train_stimulus_set, split='train')
    test_assembly = get_neuroid_assembly(neural_data_dir, test_stimulus_set, split='test')

    # Upload to S3 and save hash info to json
    print("Packaging and uploading data to S3...")
    print("Uploading train stimuli...")
    stim_train_info = package_stimulus_set(
        catalog_name=None,  # catalogs are deprecated
        proto_stimulus_set=train_stimulus_set,
        stimulus_set_identifier=train_stimulus_set.name,
        bucket_name=bucket_name
    )
    print("Uploading test stimuli...")
    stim_test_info = package_stimulus_set(
        catalog_name=None,  # catalogs are deprecated
        proto_stimulus_set=test_stimulus_set,
        stimulus_set_identifier=test_stimulus_set.name,
        bucket_name=bucket_name
    )
    print('Hashes and ids of StimulusSets:')
    print(stim_train_info)
    print(stim_test_info)

    # print("Uploading train neural data assembly... (can take over an hour)")
    # assy_train_info = package_data_assembly(
    #     catalog_identifier=None,  # catalogs are deprecated
    #     proto_data_assembly=train_assembly,
    #     assembly_identifier=train_assembly.name,
    #     stimulus_set_identifier=train_stimulus_set.name,
    #     assembly_class_name="NeuroidAssembly",
    #     bucket_name=bucket_name,
    # )

    # print("Uploading test neural data assembly... (ca. 20 min)")
    # assy_test_info = package_data_assembly(
    #     catalog_identifier=None,  # catalogs are deprecated
    #     proto_data_assembly=test_assembly,
    #     assembly_identifier=test_assembly.name,
    #     stimulus_set_identifier=test_stimulus_set.name,
    #     assembly_class_name="NeuroidAssembly",
    #     bucket_name=bucket_name,
    # )
    # print('Hashes and ids of Assemblies:')
    # print(assy_train_info)
    # print(assy_test_info)

    # Save the info jsons to the file system
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'stim_train_info.json', 'w') as f:
        json.dump(stim_train_info, f)
    with open(output_path / 'stim_test_info.json', 'w') as f:
        json.dump(stim_test_info, f)
    # with open(output_path / 'assy_train_info.json', 'w') as f:
    #     json.dump(assy_train_info, f)
    # with open(output_path / 'assy_test_info.json', 'w') as f:
    #     json.dump(assy_test_info, f)
    print(f"Saved stimulus and assembly info jsons to {output_path}")
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Package THINGS EEG2 data with time bins")
    parser.add_argument(
        '--neural-data-dir', '--neuro-dir',
        dest='neural_data_dir',
        type=str,
        required=True,
        help='Path to the neural data directory containing preprocessed EEG .npy files.'
    )
    parser.add_argument(
        '--things-image-dir', '--imgs-dir',
        dest='things_image_dir',
        type=str,
        required=True,
        help='Path to the THINGS image database directory containing images.'
    )
    parser.add_argument(
        '--bucket-name',
        dest='bucket_name',
        type=str,
        default=DEFAULT_BUCKET_NAME,
        help='The address of the S3 bucket where the files are uploaded.'
    )
    parser.add_argument(
        '--output-path',
        dest='output_path',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help='Directory where the file hashes are saved.'
    )
    parser.add_argument(
        '--cache-dir',
        dest='cache_dir',
        type=str,
        default=None,
        help='Optional directory path to use as the Brainio cache location.'
    )
    args = parser.parse_args()
    
    # tells brainio where to put the files the package methods create before uploading
    if args.cache_dir:
        os.environ['BRAINIO_HOME'] = args.cache_dir
    
    neural_data_dir = Path(args.neural_data_dir)
    things_image_dir = Path(args.things_image_dir)
    bucket_name = args.bucket_name
    output_path = Path(args.output_path)

    package_data(neural_data_dir, things_image_dir, bucket_name, output_path)