import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Union, Tuple, Dict, List
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.packaging import package_data_assembly, package_stimulus_set

#####
# Package the Hebart et al. (2023) fMRI dataset based on THINGS images
# neuroids: 3 subjects with varying numbers of voxels in each region
# train stimuli: 8640 presentations (720 categories x 12 images x 1 repetition)
# test stimuli: 1200 presentations (100 images x 12 repetitions)
# time bins: 1
# 
# this packaging script provides the option to filter for reliable voxels based on a noise ceiling threshold
# on brainscore we uploaded all voxels to give reasearchers maximum flixibility
#####


DEFAULT_BUCKET_NAME = "brainscore-storage/brainio-brainscore"
DEFAULT_OUTPUT_PATH = "./uploaded_data_info/"

SPLITS = ['train', 'test']
ROIS = {
    'V1': ['V1'],
    'V2': ['V2'],
    'V4': ['hV4'],
    'IT': [
        'glasser-TE1p', 'glasser-TE2p', 'glasser-FFC', 'glasser-VVC', 'glasser-VMV2', 
        'glasser-VMV3', 'glasser-PHA1', 'glasser-PHA2', 'glasser-PHA3'
    ]
}
REPS = {'train': 1, 'test': 12}
SUBJECTS = ["01", "02", "03"]
METADATA_COLUMNS = ['voxel_id', 'voxel_x', 'voxel_y', 'voxel_z',
                    'nc_singletrial', 'nc_testset', 'splithalf_uncorrected', 'splithalf_corrected',
                    'prf-eccentricity', 'prf-polarangle', 'prf-rsquared', 'prf-size']

DEFAULT_FILTER = False  # Set to True to filter voxels by noise ceiling threshold
DEFAULT_NOISE_CEILING_THRESHOLD = 0.3  # Voxels that do not pass this threshold will be excluded (if FILTER_VOXELS=True)

def load_subject_data(
    fmri_data_dir: Union[str, Path], 
    subject_id: str, 
    drop_voxel_id_from_responses: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the fMRI data and associated metadata for a given subject.
    """
    if not isinstance(fmri_data_dir, Path):
        fmri_data_dir = Path(fmri_data_dir)
        
    if not str(fmri_data_dir).endswith('/betas_csv'):
        fmri_data_dir = fmri_data_dir / "betas_csv"
    
    # Load the metadata
    metadata_file = fmri_data_dir / f"sub-{subject_id}_StimulusMetadata.csv"
    metadata = pd.read_csv(metadata_file)

    # Load the fMRI data
    responses_file = fmri_data_dir / f"sub-{subject_id}_ResponseData.h5"
    responses = pd.read_hdf(responses_file)
    if drop_voxel_id_from_responses:
        responses = responses.drop(columns=['voxel_id'])
    
    # Load the voxel data
    voxel_data_file = fmri_data_dir / f"sub-{subject_id}_VoxelMetadata.csv"
    voxel_data = pd.read_csv(voxel_data_file)

    return responses, metadata, voxel_data


def process_subject_data(
    responses: pd.DataFrame, 
    metadata: pd.DataFrame, 
    voxel_data: pd.DataFrame, 
    rois: Dict[str, List[str]], 
    noise_ceiling_threshold: float = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, pd.DataFrame]]:
    """
    Process the fMRI data for a given subject.
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

    # Get the train and test responses
    train_responses, test_responses, neuroid_metadata = {}, {}, {}
    for roi_name, roi_list in rois.items():
        # Get the voxel data for the current ROI
        roi_mask = voxel_data[roi_list].sum(axis=1).values.astype(bool)
        # check if any voxel is assigned to multiple ROIs
        assert np.all(voxel_data[roi_list].sum(axis=1).values <= 1), f"Voxel assigned to multiple ROIs in {roi_name}"

        roi_data = responses[roi_mask].to_numpy().T
        noise_ceiling = voxel_data.loc[roi_mask, 'nc_testset'].to_numpy()

        neuroid_metadata_roi = voxel_data.loc[roi_mask, METADATA_COLUMNS]
        precise_rois = voxel_data.loc[roi_mask, roi_list].idxmax(axis=1).values
        assert len(precise_rois) == len(neuroid_metadata_roi), "Mismatch in precise ROIs and neuroid metadata lengths"
        neuroid_metadata_roi['roi'] = precise_rois

        # Get the train and test responses for the current ROI
        train_responses_roi = roi_data[train_indices]
        test_responses_roi = roi_data[test_indices]
        
        # If a noise ceiling threshold is provided, filter out the voxels
        # that do not meet the threshold
        if noise_ceiling_threshold is not None:
            voxel_mask = noise_ceiling > noise_ceiling_threshold * 100
            train_responses_roi = train_responses_roi[:, voxel_mask]
            test_responses_roi = test_responses_roi[:, voxel_mask]
            neuroid_metadata_roi = neuroid_metadata_roi[voxel_mask]
            assert len(neuroid_metadata_roi) == train_responses_roi.shape[1], \
                "Mismatch between meta data and responses after voxel filtering"
            # print the percentage of voxels kept
            print(f"ROI {roi_name}: kept {np.sum(voxel_mask)} / {len(voxel_mask)} voxels ({100 * np.sum(voxel_mask) / len(voxel_mask):.2f}%)")
        else:
            print(f"ROI {roi_name}: kept all {len(noise_ceiling)} voxels (no filtering applied)")
        
        train_responses[roi_name] = train_responses_roi
        test_responses[roi_name] = test_responses_roi
        neuroid_metadata[roi_name] = neuroid_metadata_roi

    return train_responses, test_responses, train_stimuli, test_stimuli, neuroid_metadata


def get_stimulus_set(
    things_image_dir: Path, 
    image_paths: List[str], 
    split: str
) -> StimulusSet:
    """
    Load and transform the training images
    """

    stimulus_ids, stimulus_paths, labels = [], {}, {}

    for img_path in tqdm(image_paths, total=len(image_paths), desc=f"Loading {split} stimuli"):
        stimulus_id = img_path.split('.')[0]
        img_class = stimulus_id.rsplit('_', 1)[0]  # Get the class from the stimulus ID
        stimulus_path = things_image_dir / img_class / img_path
        
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
    stimulus_set.name = f'THINGS_fMRI_{split}_Stimuli'
    stimulus_set.identifier = f'things_fMRI_{split}_stimuli'

    return stimulus_set

def get_neuroid_assembly(neural_responses: Dict[str, Dict[str, np.ndarray]],
                        neuroid_metadata:Dict[str, Dict[str, pd.DataFrame]],
                        stimulus_set: StimulusSet,
                        split: str) -> NeuroidAssembly:
    data_neural_concat = []
    subject_indices, roi_indices, neuroid_indices = [], [], []
    metadata_concat = {col: [] for col in METADATA_COLUMNS + ['roi']} # each entry will be a list for selected metadata based on METADATA_COLUMNS

    for subject in SUBJECTS:
        total_neuroid_per_subject = 0
        for roi_name in ROIS.keys():
            neural_responses_roi = neural_responses[subject][roi_name]
            neuroid_metadata_roi = neuroid_metadata[subject][roi_name]
            
            mean = neural_responses_roi.mean(axis=0)
            std = neural_responses_roi.std(axis=0)
            neural_responses_roi = (neural_responses_roi - mean) / std
            
            data_neural_concat.append(neural_responses_roi)
            
            n_neuroids = neural_responses_roi.shape[1]

            subject_indices.extend([subject] * n_neuroids)
            roi_indices.extend([roi_name] * n_neuroids)
            for col in METADATA_COLUMNS + ['roi']:
                assert col in neuroid_metadata_roi.columns, f"Column {col} not found in neuroid metadata"
                metadata_concat[col].extend(neuroid_metadata_roi[col].tolist())
                assert len(metadata_concat[col]) == len(neuroid_indices) + total_neuroid_per_subject + n_neuroids, f"Mismatch in metadata length for column {col}"


            total_neuroid_per_subject += n_neuroids
        
        neuroid_indices.extend(range(total_neuroid_per_subject))

    data_neural_concat = np.concatenate(data_neural_concat, axis=1)
    data_neural_flatten = data_neural_concat.reshape(data_neural_concat.shape[0], -1)
    
    repetition_indices = np.tile(np.arange(REPS[split]), data_neural_concat.shape[0] // REPS[split])
    
    assembly = NeuroidAssembly(
        data_neural_flatten.reshape(
            data_neural_concat.shape[0], data_neural_concat.shape[1], 1),
        dims=["presentation", "neuroid", "time_bin"],
        coords={
            "stimulus_id": ("presentation", np.repeat(stimulus_set["stimulus_id"].values, REPS[split])),
            "stimulus_label": ("presentation", np.repeat(stimulus_set["label"].values, REPS[split])),
            "object_name": ("presentation", np.repeat(stimulus_set["object_name"].values, REPS[split])),
            "stimulus_label_idx": ("presentation", np.repeat(stimulus_set["label_idx"].values, REPS[split])),
            "repetition": ("presentation", repetition_indices),
            "region": ("neuroid", roi_indices),
            "subject": ("neuroid", subject_indices),
            "neuroid_id": ("neuroid", neuroid_indices),
            **{col: ("neuroid", metadata_concat[col]) for col in METADATA_COLUMNS + ['roi']},
            "time_bin_start": ("time_bin", [70]),
            "time_bin_end": ("time_bin", [170])
        },
    )

    # Assign a name to the data assembly
    assembly.name = f'THINGS_fMRI_{split}_Assembly'
    assembly.attrs['stimulus_set'] = stimulus_set


    # Ensure the assembly's stimulus IDs match those provided in the stimulus set
    assert np.array_equal(
        assembly["stimulus_id"].values, np.repeat(stimulus_set["stimulus_id"].values, REPS[split])
    ), "Stimulus IDs do not match."

    return assembly

def package_data(
    neural_data_dir: Union[str, Path], 
    things_image_dir: Union[str, Path], 
    bucket_name: str, 
    output_path: Union[str, Path],
    filter_voxels: bool,
    noise_ceiling_threshold: float,
) -> None:
    """
    Main function to package THINGS fMRI data for Brain-Score.
    
    Args:
        neural_data_dir: Path to the directory containing fMRI data
        things_image_dir: Path to the THINGS image database directory
        bucket_name: Name of the S3 bucket for uploading packaged data
        output_path: Directory path where file hashes will be saved
        filter_voxels: Whether to filter voxels based on noise ceiling threshold
        noise_ceiling_threshold: The noise ceiling threshold to use for filtering
    """
    neural_data_dir = Path(neural_data_dir)
    things_image_dir = Path(things_image_dir)
    output_path = Path(output_path)
        
        
    responses_subjects, metadata_subjects, voxdata_subjects = {}, {}, {}
    for subject_id in tqdm(SUBJECTS, desc="Loading subject data"):
        responses, metadata, voxdata = load_subject_data(neural_data_dir, subject_id)
        responses_subjects[subject_id] = responses
        metadata_subjects[subject_id] = metadata
        voxdata_subjects[subject_id] = voxdata
    
    # Process all subject data
    train_responses_subjects, test_responses_subjects = {}, {}
    train_stimuli, test_stimuli = None, None
    neuroid_metadata_subjects = {} 

    for subject_id in tqdm(SUBJECTS, desc="Processing subject data"):
        responses = responses_subjects[subject_id]
        metadata = metadata_subjects[subject_id]
        voxel_data = voxdata_subjects[subject_id]
        
        # Verify stimulus filename format: {concept}_id.jpg
        stimuli = metadata['stimulus'].values
        concepts = metadata['concept'].values
        assert all(
            stimulus.rsplit('_', 1)[0] == concept 
            for stimulus, concept in zip(stimuli, concepts)
        ), f"Stimulus format mismatch for subject {subject_id}"
        
        # Process the data for the current subject
        train_responses, test_responses, train_stim, test_stim, neuroid_metadata = process_subject_data(
            responses, metadata, voxel_data, ROIS, 
            noise_ceiling_threshold=noise_ceiling_threshold if filter_voxels else None
        )
        
        train_responses_subjects[subject_id] = train_responses
        test_responses_subjects[subject_id] = test_responses
        neuroid_metadata_subjects[subject_id] = neuroid_metadata

        if train_stimuli is None:
            train_stimuli = train_stim
            test_stimuli = test_stim
        else:
            assert np.array_equal(train_stimuli, train_stim), \
                f"Train stimuli mismatch for subject {subject_id}"
            assert np.array_equal(test_stimuli, test_stim), \
                f"Test stimuli mismatch for subject {subject_id}"
    
    # Call functions to create the train stimulus set and assembly
    train_stimulus_set = get_stimulus_set(things_image_dir, train_stimuli, 'train')
    train_assembly = get_neuroid_assembly(train_responses_subjects, neuroid_metadata_subjects, train_stimulus_set, 'train')
    
    # Call functions to create the test stimulus set and assembly
    test_stimulus_set = get_stimulus_set(things_image_dir, test_stimuli, 'test')
    test_assembly = get_neuroid_assembly(test_responses_subjects, neuroid_metadata_subjects, test_stimulus_set, 'test')
    
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
    
    print("Uploading train neural data assembly...")
    assy_train_info = package_data_assembly(
        catalog_identifier=None,  # catalogs are deprecated
        proto_data_assembly=train_assembly,
        assembly_identifier=train_assembly.name,
        stimulus_set_identifier=train_stimulus_set.name,
        assembly_class_name="NeuroidAssembly",
        bucket_name=bucket_name,
    )
    
    print("Uploading test neural data assembly...")
    assy_test_info = package_data_assembly(
        catalog_identifier=None,  # catalogs are deprecated
        proto_data_assembly=test_assembly,
        assembly_identifier=test_assembly.name,
        stimulus_set_identifier=test_stimulus_set.name,
        assembly_class_name="NeuroidAssembly",
        bucket_name=bucket_name,
    )
    print('Hashes and ids of Assemblies:')
    print(assy_train_info)
    print(assy_test_info)
    
    # Save the info jsons to the file system
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
    parser = ArgumentParser(description="Package THINGS fMRI data")
    parser.add_argument(
        '--neural-data-dir', '--neuro-dir',
        dest='neural_data_dir',
        type=str,
        required=True,
        help='Path to the neural data directory containing fMRI .csv and .h5 files.'
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
    parser.add_argument(
        '--filter-threshold',
        dest='filter_threshold',
        type=float,
        nargs='?',
        const=DEFAULT_NOISE_CEILING_THRESHOLD,
        default=None,
        help=f'Filter voxels by noise ceiling threshold. Use --filter-threshold to apply default threshold ({DEFAULT_NOISE_CEILING_THRESHOLD}), '
             f'or --filter-threshold 0.4 to specify a custom threshold. Omit to include all voxels.'
    )
    args = parser.parse_args()
    
    # tells brainio where to put the files the package methods create before uploading
    if args.cache_dir:
        os.environ['BRAINIO_HOME'] = args.cache_dir
    
    if args.filter_threshold is not None:
        filter_voxels = True
        noise_ceiling_threshold = args.filter_threshold
        print(f"Filtering enabled with noise ceiling threshold: {args.filter_threshold}")
    else:
        filter_voxels = DEFAULT_FILTER # use the default if no argument is given
        noise_ceiling_threshold = DEFAULT_NOISE_CEILING_THRESHOLD
        print("Filtering disabled - including all voxels")
    
    neural_data_dir = Path(args.neural_data_dir)
    things_image_dir = Path(args.things_image_dir)
    bucket_name = args.bucket_name
    output_path = Path(args.output_path)
    
    package_data(neural_data_dir, things_image_dir, bucket_name, output_path, filter_voxels, noise_ceiling_threshold)
