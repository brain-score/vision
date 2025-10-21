from argparse import ArgumentParser
import os
from pathlib import Path
import json

import numpy as np
from tqdm.auto import tqdm

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.packaging import package_data_assembly, package_stimulus_set

DEFAULT_BUCKET_NAME = "brainscore-storage/brainio-brainscore"
DEFAULT_OUTPUT_PATH = "./uploaded_data_info/"

SUBJECTS = ["sub-{:02d}".format(i) for i in range(1, 11)]
SPLITS = ['train', 'test']


def package_everything(neural_data_dir: Path, things_image_db_dir: Path, bucket_name: str, output_path: Path) -> None:
    """
    Args:
        neural_data_dir: Path to the neural data directory containing preprocessed EEG .npy files
        things_image_db_dir: Path to the stimulus data directory containing images
        bucket_name: S3 bucket name for upload
        output_path: Directory path where file hashes will be saved
    """
    image_metadata_path = neural_data_dir / "image_metadata.npy"
    image_metadata = np.load(image_metadata_path, allow_pickle=True, fix_imports=True).item()

    print(f"Number of training stimuli: {len(image_metadata['train_img_files'])}, Number of testing stimuli: {len(image_metadata['test_img_files'])}")
    print(f"Example stimulus file: {image_metadata['train_img_files'][0]}, concept: {image_metadata['train_img_concepts'][0]}")

    #check that the images really are at the given path
    for split in SPLITS:
        file_name = image_metadata[f'{split}_img_files'][0]
        stimulus_id = file_name.split('.')[0]
        img_class = stimulus_id.rsplit('_', 1)[0]  # Get the class from the stimulus ID
        check_path = things_image_db_dir / img_class / file_name
        assert check_path.exists(), f"No image at: {check_path}"

    # get the stimuli to populate each StimulusSet
    for split in SPLITS:
        file_names = image_metadata[f'{split}_img_files']
        
        stimulus_ids, stimulus_paths, labels = [], {}, {}
        for file_name in tqdm(file_names, total=len(file_names)):

            stimulus_id = file_name.split('.')[0]
            img_class = stimulus_id.rsplit('_', 1)[0]  # Get the class from the stimulus ID
            stimulus_path = things_image_db_dir / img_class / file_name
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

        globals()[f'stimulus_set_{split}'] = stimulus_set
        del stimulus_set

    # neural data to populate each NeuroidAssembly
    training_subj_data = np.load(neural_data_dir / "sub-01" / "preprocessed_eeg_training.npy", allow_pickle=True).item()
    testing_subj_data = np.load(neural_data_dir / "sub-01" / "preprocessed_eeg_test.npy", allow_pickle=True).item()
    print(f"time bins are shaped: {training_subj_data['times'].shape}, times: {training_subj_data['times'][:3]} ...")
    print(f"there are {len(training_subj_data['ch_names'])} channels: {training_subj_data['ch_names'][:3]} ...")
    print(f"preprocessed train EEG data (4 repetitions of each stimulus) has shape {training_subj_data['preprocessed_eeg_data'].shape}")
    print(f"preprocessed test EEG data (80 repetitions of each stimulus) has shape {testing_subj_data['preprocessed_eeg_data'].shape}")

    for split in SPLITS:
        print(f"Processing {split} split across all subjects...")
        if split == 'train':
            filename = "preprocessed_eeg_training.npy"
        elif split == 'test':
            filename = "preprocessed_eeg_test.npy"
        else:
            raise ValueError(f"Unknown split: {split}")
        data_neural_concat = []
        subject_names, channel_names, recoding_times, neuroid_indices = [], [], [], []
        for subj in tqdm(SUBJECTS, total=len(SUBJECTS)):
            neural_data = np.load(neural_data_dir / subj / filename, allow_pickle=True).item()

            subject_responses = neural_data['preprocessed_eeg_data']
            ch_names = neural_data['ch_names']
            times = neural_data['times']
            n_stimuli, n_trials, n_channels, n_time_points = subject_responses.shape

            subject_names.extend([subj] * n_channels * n_time_points)
            channel_names.extend(np.repeat(ch_names, n_time_points))
            recoding_times.extend(np.tile(times, n_channels))
            neuroid_indices.extend(np.arange(n_channels * n_time_points))

            # Z-score normalization across trials and stimuli
            # subject_responses: # shape: (n_stimuli, n_trials, n_channels, n_time_points)
            mean = np.mean(subject_responses, axis=(0, 1))
            std = np.std(subject_responses, axis=(0, 1))
            subject_responses = (subject_responses - mean) / std 
            
            subject_responses = subject_responses.reshape(n_stimuli * n_trials, n_channels * n_time_points)
            if subj == 'sub-01': 
                print(f"each subjects recordings have data shape {subject_responses.shape} after reshaping")
            data_neural_concat.append(subject_responses)

        ## Concatenate across subjects
        data_neural_concat = np.concatenate(data_neural_concat, axis=1)
        trial_indices = np.tile(np.arange(n_trials), n_stimuli)
        print(f"Concatenated neural data shape for split {split}: {data_neural_concat.shape}"+\
              f"(#presentations = {n_stimuli} Stimuli x {n_trials} Trials,"+\
              f" #features = {n_channels} Channels x {n_time_points} Timepoints x {len(SUBJECTS)} Subjects)")

        total_features = len(SUBJECTS) * n_channels * n_time_points  # 10 subjects * 17 channels * 100 timepoints
        total_presentations = n_stimuli * n_trials  # 22248 stimuli * 4 or 80 trials
        assert np.all([
            len(subject_names) == len(channel_names) == len(recoding_times) == len(neuroid_indices) == data_neural_concat.shape[1] == total_features,
            len(trial_indices) == data_neural_concat.shape[0] == total_presentations
        ])

        stimulus_set = globals()[f'stimulus_set_{split}']
        assembly = NeuroidAssembly(
            data_neural_concat.reshape(
                data_neural_concat.shape[0], data_neural_concat.shape[1], 1),
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
                "recoding_time": ("neuroid", recoding_times),
                "time_bin_start": ("time_bin", [-20]),
                "time_bin_end": ("time_bin", [1000])
            },
        )

        # Assign a name to the data assembly
        assembly.name = f'THINGS_EEG2_{split}_Assembly'
        assembly.attrs['stimulus_set'] = stimulus_set

        globals()[f'assembly_{split}'] = assembly
        del assembly

    stimulus_set_train = globals()['stimulus_set_train']
    stimulus_set_test = globals()['stimulus_set_test']
    assembly_train = globals()['assembly_train']
    assembly_test = globals()['assembly_test']

    print("Brainio is now packaging the stimuli for upload. Don't worry, this might take a moment")
    print("Uploading train stimuli...")
    stim_train_info = package_stimulus_set(
                            catalog_name=None,  # catalogs are deprecated
                            proto_stimulus_set=stimulus_set_train,
                            stimulus_set_identifier=stimulus_set_train.name,
                            bucket_name=bucket_name)
    print("Uploading test stimuli...")
    stim_test_info = package_stimulus_set(
                            catalog_name=None,  # catalogs are deprecated
                            proto_stimulus_set=stimulus_set_test,
                            stimulus_set_identifier=stimulus_set_test.name,
                            bucket_name=bucket_name)
    print('Hashes and ids of StimulusSets:')
    print(stim_train_info)
    print(stim_test_info)

    print("Uploading train neural data assembly... (can take over an hour)")
    assy_train_info = package_data_assembly(
            catalog_identifier=None,  # catalogs are deprecated
            proto_data_assembly=assembly_train, 
            assembly_identifier=assembly_train.name,
            stimulus_set_identifier=stimulus_set_train.name,
            assembly_class_name="NeuroidAssembly",
            bucket_name=bucket_name,
        )

    print("Uploading test neural data assembly... (ca. 20 min)")
    assy_test_info = package_data_assembly(
            catalog_identifier=None,  # catalogs are deprecated
            proto_data_assembly=assembly_test, 
            assembly_identifier=assembly_test.name,
            stimulus_set_identifier=stimulus_set_test.name,
            assembly_class_name="NeuroidAssembly",
            bucket_name=bucket_name,
        )
    print('Hashes and ids of Assemblies:')
    print(assy_train_info)
    print(assy_test_info)

    # save the info jsons to the file system
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'stim_train_info.json', 'w') as f:
        json.dump(stim_train_info, f)
    with open(output_path / 'stim_test_info.json', 'w') as f:
        json.dump(stim_test_info, f)
    with open(output_path / 'assy_train_info.json', 'w') as f:
        json.dump(assy_train_info, f)
    with open(output_path / 'assy_test_info.json', 'w') as f:
        json.dump(assy_test_info, f)
    print(f"Saved stimulus and assembly info jsons to {output_path}")
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Package THINGS EEG2 data")
    parser.add_argument(
        '--neural-data-dir', '--neuro-dir',
        dest='neural_data_dir',
        type=str,
        required=True,
        help='Path to the neural data directory containing preprocessed EEG .npy files.'
    )
    parser.add_argument(
        '--things-image-db-dir', '--imgs-dir',
        dest='things_image_db_dir',
        type=str,
        required=True,
        help='Path to the stimulus data directory containing images.'
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
    things_image_db_dir = Path(args.things_image_db_dir)
    bucket_name = args.bucket_name
    output_path = Path(args.output_path)

    package_everything(neural_data_dir, things_image_db_dir, bucket_name, output_path)