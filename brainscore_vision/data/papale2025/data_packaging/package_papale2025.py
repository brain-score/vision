from argparse import ArgumentParser
import numpy as np
from tqdm.auto import tqdm
import os
from pathlib import Path
from collections import defaultdict
import json
from typing import Union

# do not import mat73 NOT HERE -> CAUSES RuntimeError: NetCDF: HDF error if imported before core modules

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.packaging import package_data_assembly, package_stimulus_set

# Here importing mat73 is fine:
import mat73

#####
# Package the Papale et al. (2025) TVSD dataset based on THINGS images
# neuroids: 2 subjects with varying numbers of reliable electrodes in each region
# train stimuli: 22248 presentations (1854 categories x 12 images x 1 repetition)
# test stimuli: 3000 presentations (100 images x 30 repetitions)
# time bins: 1 (we take the averaged MUA provided by the authors)
# 
# this packaging script provides the option to filter for reliable voxels based on electrode reliability
# on brainscore we uploaded all voxels to give reasearchers maximum flixibility
# 
# 
# ATTENTION: The stimuli where packaged at 500x500 pixel resolution as they were shown in the original experiment.
# Use the create_THINGS_resized.py script to create a copy of the THINGS database with the correct resolution.
# Set --things-image-dir to the path of the resized THINGS images.
#####


DEFAULT_BUCKET_NAME = "brainscore-storage/brainio-brainscore" 
DEFAULT_OUTPUT_PATH = "./uploaded_data_info/"

SPLITS = ['train', 'test']
REGIONS = ["V1", "V4", "IT"]
REGION_CHANNELS_MAP = {
    "monkeyN": {
        "V1": (0, 511),
        "V4": (512, 767),
        "IT": (768, 1023),
    },
    "monkeyF": {
        "V1": (0, 511),
        "V4": (832, 1023),
        "IT": (512, 831),
    }
}
REPS = {'train': 1, 'test': 30}
SUBJECTS = ['monkeyF', 'monkeyN']

DEFAULT_FILTER = False  # Set to True to filter electrodes by reliability threshold
DEFAULT_RELIABILITY_THRESHOLD = 0.3  # Electrodes that do not pass this threshold will be excluded


def get_stimulus_set(image_metadata: dict, things_image_dir: Path, split: str) -> StimulusSet:
    """
    Build the StimulusSet for the given split.
    
    Args:
        image_metadata: Dictionary containing metadata for stimuli
        things_image_dir: Path to the stimulus directory containing images
        split: 'train' or 'test'
    
    Returns:
        StimulusSet for the specified split
    """
    
    # Check that the images really are at the given path
    check_path_str = image_metadata[f'{split}_imgs']['things_path'][0].replace('\\', '/')
    check_path = things_image_dir / check_path_str
    assert check_path.exists(), f"No image at: {check_path}, please correct the image path"

    # Get the stimuli to populate each StimulusSet
    stimulus_ids, stimulus_paths, labels = [], {}, {}

    img_classes = image_metadata[f'{split}_imgs']['class']
    img_paths = image_metadata[f'{split}_imgs']['things_path']
    for img_class, img_path in tqdm(zip(img_classes, img_paths), total=len(img_classes)):
        
        img_path = img_path.replace('\\', '/')
        stimulus_path = things_image_dir / img_path
        stimulus_id = stimulus_path.stem.split('/')[-1]
        
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
    stimulus_set.name = f'THINGS_TVSD_{split}_Stimuli'
    stimulus_set.identifier = f'things_tvsd_{split}_stimuli'

    print(f'Created StimulusSet for {split} with {len(stimulus_set)} stimuli.')
    return stimulus_set

def get_neuroid_assembly(neural_data_dir: Path, stimulus_set: StimulusSet, split: str, filter_electrodes: bool, reliability_threshold: float) -> NeuroidAssembly:
    """
    Create the NeuroidAssembly for the given split.
    
    Args:
        neural_data_dir: Path to the neural data directory containing .mat files
        stimulus_set: StimulusSet for the split
        split: 'train' or 'test'
        filter_electrodes: Whether to filter electrodes based on reliability threshold
        reliability_threshold: The reliability threshold to use for filtering
    
    Returns:
        NeuroidAssembly for the specified split
    """

    # Neural data to populate each NeuroidAssembly
    whole_split_neural = defaultdict(dict)
    oracle_region = defaultdict(dict)
    SNR_region = defaultdict(dict)
    SNR_max_region = defaultdict(dict)
    reliability_region = defaultdict(dict)
    for subject in tqdm(SUBJECTS, desc=f"Processing {split} subjects"):
        file_path = neural_data_dir / subject / 'THINGS_normMUA.mat'
        neural_data = mat73.loadmat(file_path)
        if split == 'train':
            neural_responses = neural_data[f'{split}_MUA']
            repetitions = 1
        else:
            # Get MUA with repetitions
            neural_responses = neural_data[f'{split}_MUA_reps']
            repetitions = neural_responses.shape[2]
        
        # Metadata
        reliability = neural_data['reliab']
        SNR = neural_data['SNR'].mean(axis=1)
        SNR_max = neural_data['SNR_max']
        oracle = neural_data['oracle']
        
        electrode_reliability = reliability.mean(axis=1)
        if filter_electrodes:
            is_reliable = electrode_reliability > reliability_threshold
            assert neural_responses.shape[0] == is_reliable.shape[0], "Mismatch in number of electrodes"
        else:
            is_reliable = np.ones(electrode_reliability.shape[0], dtype=bool)  # Keep all electrodes
            assert neural_responses.shape[0] == is_reliable.shape[0], "Mismatch in number of electrodes"

        for region, (start, end) in REGION_CHANNELS_MAP[subject].items(): 
            neuroid_start, neuroid_end = start, end + 1
            neural_responses_region = neural_responses[neuroid_start:neuroid_end]
            assert neural_responses_region.shape[0] == (neuroid_end - neuroid_start), "Mismatch in region electrode count"
            neural_responses_region = neural_responses_region[is_reliable[neuroid_start:neuroid_end]]

            # Flatten repetitions
            if split == 'test':
                n_electrodes, n_stimuli, n_reps = neural_responses_region.shape
                neural_responses_region = neural_responses_region.reshape(n_electrodes, n_stimuli * n_reps)

            whole_split_neural[subject][region] = neural_responses_region.T  # Presentations will be first in the assembly
            if filter_electrodes:
                print(f"In {region} for {subject} there were {neural_responses_region.shape[0]} reliable electrodes "+\
                        f"out of {(neuroid_end - neuroid_start)} total electrodes " +\
                        f'final shape: {neural_responses_region.shape}')
            else:
                print(f"In {region} for {subject} there are {neural_responses_region.shape[0]} electrodes "+\
                        f'with shape: {neural_responses_region.shape}')

            oracle_region[subject][region] = oracle[neuroid_start:neuroid_end][is_reliable[neuroid_start:neuroid_end]]
            SNR_region[subject][region] = SNR[neuroid_start:neuroid_end][is_reliable[neuroid_start:neuroid_end]]
            SNR_max_region[subject][region] = SNR_max[neuroid_start:neuroid_end][is_reliable[neuroid_start:neuroid_end]]
            reliability_region[subject][region] = electrode_reliability[neuroid_start:neuroid_end][is_reliable[neuroid_start:neuroid_end]]
            assert len(oracle_region[subject][region]) == len(SNR_region[subject][region]) == len(SNR_max_region[subject][region]) == len(reliability_region[subject][region]) == neural_responses_region.shape[0], \
                "Mismatch in metadata lengths"

    data_neural_concat = []  # To flatten the whole_split_neural[subject][region] into neuroid x trial x time_bin
    subject_indices, region_indices, neuroid_indices = [], [], []
    oracle_concat, SNR_concat, SNR_max_concat, reliability_concat = [], [], [], []
    for subject in SUBJECTS:
        total_neuroid_per_subject = 0
        for region in REGIONS:
            neural_responses_region = whole_split_neural[subject][region]
            data_neural_concat.append(neural_responses_region)
            
            n_neuroids = neural_responses_region.shape[1]

            subject_indices.extend([subject] * n_neuroids)
            region_indices.extend([region] * n_neuroids)
            total_neuroid_per_subject += n_neuroids

            oracle_concat.append(oracle_region[subject][region])
            SNR_concat.append(SNR_region[subject][region])
            SNR_max_concat.append(SNR_max_region[subject][region])
            reliability_concat.append(reliability_region[subject][region])

        neuroid_indices.extend(range(total_neuroid_per_subject))

    data_neural_concat = np.concatenate(data_neural_concat, axis=1)
    data_neural_flatten = data_neural_concat.reshape(data_neural_concat.shape[0], -1)

    oracle_concat = np.concatenate(oracle_concat)
    SNR_concat = np.concatenate(SNR_concat)
    SNR_max_concat = np.concatenate(SNR_max_concat)
    reliability_concat = np.concatenate(reliability_concat)
    assert len(oracle_concat) == len(SNR_concat) == len(SNR_max_concat) == len(reliability_concat) \
        == data_neural_concat.shape[1] == len(neuroid_indices), f"Mismatch in metadata lengths after concatenation: {len(oracle_concat)}, {len(SNR_concat)}, {len(SNR_max_concat)}, {len(reliability_concat)}, {data_neural_concat.shape[1]}, {len(neuroid_indices)}"

    repetition_indices = np.tile(np.arange(repetitions), data_neural_concat.shape[0] // repetitions)

    assembly = NeuroidAssembly(
        data_neural_flatten.reshape(
            data_neural_concat.shape[0], data_neural_concat.shape[1], 1),
        dims=["presentation", "neuroid", "time_bin"],
        coords={
            "stimulus_id": ("presentation", np.repeat(stimulus_set["stimulus_id"].values, repetitions)),
            "stimulus_label": ("presentation", np.repeat(stimulus_set["label"].values, repetitions)),
            "object_name": ("presentation", np.repeat(stimulus_set["object_name"].values, repetitions)),
            "stimulus_label_idx": ("presentation", np.repeat(stimulus_set["label_idx"].values, repetitions)),
            "repetition": ("presentation", repetition_indices),
            "region": ("neuroid", region_indices),
            "subject": ("neuroid", subject_indices),
            "neuroid_id": ("neuroid", neuroid_indices),
            "SNR": ("neuroid", SNR_concat),
            "SNR_max": ("neuroid", SNR_max_concat),
            "reliability": ("neuroid", reliability_concat),
            "oracle": ("neuroid", oracle_concat),
            "time_bin_start": ("time_bin", [70]),
            "time_bin_end": ("time_bin", [170])
        },
    )

    # Assign a name to the data assembly
    assembly.name = f'THINGS_TVSD_{split}_Assembly'
    assembly.attrs['stimulus_set'] = stimulus_set


    # Ensure the assembly's stimulus IDs match those provided in the stimulus set
    assert np.array_equal(
        assembly["stimulus_id"].values, np.repeat(stimulus_set["stimulus_id"].values, repetitions)
    ), "Stimulus IDs do not match."
    
    return assembly

def package_data(neural_data_dir: Path, things_image_dir: Path, bucket_name: str, output_path: Path, filter_electrodes: bool, reliability_threshold: float):

    # Use subject 1's metadata as reference
    image_metadata = mat73.loadmat(neural_data_dir / SUBJECTS[0] / '_logs' / 'things_imgs.mat')
    
    # Verify metadata consistency across subjects
    for subject in SUBJECTS[1:]:
            subject_metadata = mat73.loadmat(neural_data_dir / subject / '_logs' / 'things_imgs.mat')
            for split in ['train_imgs', 'test_imgs']:
                for key in ['class', 'things_path']:
                    assert image_metadata[split][key] == subject_metadata[split][key], f"Mismatch in {split} for key: {key}"
    print("Metadata is consistent across subjects.")

    train_stimulus_set = get_stimulus_set(image_metadata, things_image_dir, 'train')
    test_stimulus_set = get_stimulus_set(image_metadata, things_image_dir, 'test')
    train_assembly = get_neuroid_assembly(neural_data_dir, train_stimulus_set, 'train', filter_electrodes, reliability_threshold)
    test_assembly = get_neuroid_assembly(neural_data_dir, test_stimulus_set, 'test', filter_electrodes, reliability_threshold)

    print("Brainio is now packaging the stimuli for upload. Don't worry, this might take a moment")
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

    # print("Uploading train neural data assembly...")
    # assy_train_info = package_data_assembly(
    #     catalog_identifier=None,  # catalogs are deprecated
    #     proto_data_assembly=train_assembly,
    #     assembly_identifier=train_assembly.name,
    #     stimulus_set_identifier=train_stimulus_set.name,
    #     assembly_class_name="NeuroidAssembly",
    #     bucket_name=bucket_name,
    # )

    # print("Uploading test neural data assembly...")
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
    parser = ArgumentParser(description="Package TVSD data")
    parser.add_argument(
        '--neural-data-dir', '--neuro-dir', 
        dest='neural_data_dir', 
        type=str, 
        required=True,
        help='Path to the neural data directory containing .mat files.'
    )
    parser.add_argument(
        '--things-image-dir', '--imgs-dir', 
        dest='things_image_dir', 
        type=str, 
        required=True,
        help='Path to the stimulus data directory containing images.'
    )
    parser.add_argument(  # Optional bucket
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
    parser.add_argument(
        '--filter-threshold',
        dest='filter_threshold',
        type=float,
        nargs='?',
        const=DEFAULT_RELIABILITY_THRESHOLD,
        default=None,
        help=f'Filter electrodes by reliability threshold. Use --filter-threshold to apply default threshold ({DEFAULT_RELIABILITY_THRESHOLD}), '
             f'or --filter-threshold 0.4 to specify a custom threshold. Omit to include all electrodes.'
    )
    args = parser.parse_args()
    
    # tells brainio where to put the files the package methods create before uploading
    if args.cache_dir:
        os.environ['BRAINIO_HOME'] = args.cache_dir

    if args.filter_threshold is not None:
        filter_electrodes = True
        reliability_threshold = args.filter_threshold 
        print(f"Filtering enabled with reliability threshold: {args.filter_threshold}")
    else:
        filter_electrodes = DEFAULT_FILTER
        reliability_threshold = DEFAULT_RELIABILITY_THRESHOLD
        print("Filtering disabled - including all electrodes")

    neural_data_dir = Path(args.neural_data_dir)
    things_image_dir = Path(args.things_image_dir)
    bucket_name = args.bucket_name
    output_path = Path(args.output_path)

    package_data(neural_data_dir, things_image_dir, bucket_name, output_path, filter_electrodes, reliability_threshold)
