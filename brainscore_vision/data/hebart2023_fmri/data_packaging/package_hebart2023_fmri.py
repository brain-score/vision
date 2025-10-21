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

DEFAULT_BUCKET_NAME = "brainscore-storage/brainio-brainscore"
DEFAULT_OUTPUT_PATH = "./uploaded_data_info/"

SPLITS = ['train', 'test']
ROIS = {
    'V1': ['V1'],
    'V2': ['V2'],
    'V4': ['hV4'],
    # 'IT': ['IT']
    'IT': [
        'glasser-TE1p','glasser-TE2p', 'glasser-FFC', 'glasser-VVC', 'glasser-VMV2', 
        'glasser-VMV3', 'glasser-PHA1', 'glasser-PHA2', 'glasser-PHA3'
    ]
}
SUBJECTS = ["01", "02", "03"]
NOISE_CEILING_THRESHOLD = 0.3

def load_subject_data(fmri_data_dir: Union[str, Path], sub: str, drop_voxel_id_from_responses=True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the fMRI data and associated metadata for a given subject.
    """
    if not isinstance(fmri_data_dir, Path):
        fmri_data_dir = Path(fmri_data_dir)
        
    if not str(fmri_data_dir).endswith('/betas_csv'):
        fmri_data_dir = fmri_data_dir / "betas_csv"
    
    # Load the metadata
    metadata_file = fmri_data_dir / f"sub-{sub}_StimulusMetadata.csv"
    metadata = pd.read_csv(metadata_file)

    # Load the fMRI data
    responses_file = fmri_data_dir / f"sub-{sub}_ResponseData.h5"
    responses = pd.read_hdf(responses_file)
    if drop_voxel_id_from_responses:
        responses = responses.drop(columns=['voxel_id'])
    
    # Load the voxel data
    voxdata_file = fmri_data_dir / f"sub-{sub}_VoxelMetadata.csv"
    voxdata = pd.read_csv(voxdata_file)

    return responses, metadata, voxdata


def process_subject_data(responses: pd.DataFrame, metadata: pd.DataFrame, voxdata: pd.DataFrame, ROIs:Dict[str, List[str]], noise_ceiling_theshold:float=None) \
    -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Process the fMRI data for a given subject.
    """

    # Get sorting indices w.r.t. stimulus filenames
    # and split the data into train and test sets
    indices = metadata['stimulus'].argsort()
    train_mask = ((metadata['trial_type'] == 'train').values)[indices]
    train_indices, val_indices = indices[train_mask], indices[~train_mask]
    
    # Get the train and test stimuli
    train_stimuli = metadata['stimulus'][train_indices].values
    test_stimuli = metadata['stimulus'][val_indices].values
    test_stimuli = test_stimuli.reshape(-1, 12)
    
    assert np.all(test_stimuli == test_stimuli[:, [0]])
    test_stimuli = test_stimuli[:, 0]

    # Get the train and test responses
    train_responses, test_responses = {}, {}
    for roi_name, roi_list in ROIs.items():
        # Get the voxel data for the current ROI
        roimask = voxdata[roi_list].sum(axis=1).values.astype(bool)
        roidata = responses[roimask].to_numpy().T
        noise_ceiling = voxdata.loc[roimask, 'nc_testset'].to_numpy()

        # Get the train and test responses for the current ROI
        train_responses_roi = roidata[train_indices]
        test_responses_roi = roidata[val_indices]
        
        # If a noise ceiling threshold is provided, filter out the voxels
        # that do not meet the threshold
        if noise_ceiling_theshold:
            voxel_mask = noise_ceiling > noise_ceiling_theshold * 100
            train_responses_roi = train_responses_roi[:, voxel_mask]
            test_responses_roi = test_responses_roi[:, voxel_mask]
            
        train_responses[roi_name] = train_responses_roi
        test_responses[roi_name] = test_responses_roi

    return train_responses, test_responses, train_stimuli, test_stimuli


def load_stimulus_set(things_image_db_dir:Path, image_paths:List[str], split:str) -> StimulusSet:
    """
    Load and transform the training images
    """

    stimulus_ids, stimulus_paths, labels = [], {}, {}

    for img_path in tqdm(image_paths, total=len(image_paths)):
        
        stimulus_id =  img_path.split('.')[0]
        img_class = stimulus_id.rsplit('_', 1)[0]  # Get the class from the stimulus ID
        stimulus_path = things_image_db_dir / img_class / img_path
        
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
            'object_name':  labels[stimulus_id],
            'label_idx': labels_idx[stimulus_id],
        })
        
    stimulus_set = StimulusSet(stimuli)
    stimulus_set.stimulus_paths = stimulus_paths
    stimulus_set.name = f'THINGS_fRMRI_{split}_Stimuli'
    stimulus_set.identifier = f'things_tfmri_{split}_stimuli'
    # assert len(stimulus_set) == 22248

    return stimulus_set

def load_brain_data(neural_responses:Dict[str, Dict[str, np.ndarray]], stimulus_set:StimulusSet, split:str) -> NeuroidAssembly:
    data_neural_concat = []
    subject_indices, roi_indices, neuroid_indices = [], [], []
    for subject in SUBJECTS:
        total_neuroid_per_subject = 0
        for roi_name, roi_responses  in ROIS.items():
            neural_responses_roi = neural_responses[subject][roi_name]
            
            mean = neural_responses_roi.mean(axis=0)
            std = neural_responses_roi.std(axis=0)
            neural_responses_roi = (neural_responses_roi - mean) / std
            
            data_neural_concat.append(neural_responses_roi)
            
            n_neuroids = neural_responses_roi.shape[1]

            subject_indices.extend([subject] * n_neuroids)
            roi_indices.extend([roi_name] * n_neuroids)
            total_neuroid_per_subject += n_neuroids
        
        neuroid_indices.extend(range(total_neuroid_per_subject))

    data_neural_concat = np.concatenate(data_neural_concat, axis=1)
    data_neural_flatten = data_neural_concat.reshape(data_neural_concat.shape[0], -1)
    
    REPS = {'train':1, 'test':12}
    assembly = NeuroidAssembly(
        data_neural_flatten.reshape(
            data_neural_concat.shape[0], data_neural_concat.shape[1], 1),
        dims=["presentation", "neuroid", "time_bin"],
        coords={
            "stimulus_id": ("presentation", np.repeat(stimulus_set["stimulus_id"].values, REPS[split])),
            "stimulus_label": ("presentation", np.repeat(stimulus_set["label"].values, REPS[split])),
            "object_name": ("presentation", np.repeat(stimulus_set["object_name"].values, REPS[split])),
            "stimulus_label_idx": ("presentation", np.repeat(stimulus_set["label_idx"].values, REPS[split])),
            "roi": ("neuroid", roi_indices),
            "region": ("neuroid", roi_indices),
            "subject": ("neuroid", subject_indices),
            "neuroid_id": ("neuroid", neuroid_indices),
            "voxels": ("neuroid", neuroid_indices),
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

def package_data(neural_data_dir: Path, things_image_db_dir: Path, bucket_name: str, output_path: Path) -> None:
    """
    Args:
        neural_data_dir: Path to the neural data directory
        things_image_db_dir: Path to the stimulus data directory containing images
        bucket_name: S3 bucket name for upload
        output_path: Directory path where file hashes will be saved
    """
    if isinstance(neural_data_dir, str):
        neural_data_dir = Path(neural_data_dir)
    if isinstance(things_image_db_dir, str):
        things_image_db_dir = Path(things_image_db_dir)
        
        
    responses_subjects, metadata_subjects, voxdata_subjects = {}, {}, {}
    for sub in tqdm(SUBJECTS):
        responses, metadata, voxdata = load_subject_data(neural_data_dir, sub)
        responses_subjects[sub] = responses
        metadata_subjects[sub] = metadata
        voxdata_subjects[sub] = voxdata
        
    train_responses_subjects, test_responses_subjects = {}, {}
    train_stimuli, test_stimuli = None, None

    for sub in tqdm(SUBJECTS):
        responses, metadata, voxdata = responses_subjects[sub], metadata_subjects[sub], voxdata_subjects[sub]
        # Check: stimuli filename is of the form `{concept}_id.jpg`
        stimuli, concepts = metadata['stimulus'].values, metadata['concept'].values
        assert all([stimulus.rsplit('_', 1)[0] == concept for stimulus, concept in zip(stimuli, concepts)])
        
        # Process the data for the current subject
        train_responses_, test_responses_, train_stimuli_, test_stimuli_ = \
            process_subject_data(responses, metadata, voxdata, ROIS, noise_ceiling_theshold=NOISE_CEILING_THRESHOLD)
        
        train_responses_subjects[sub] = train_responses_
        test_responses_subjects[sub] = test_responses_

        if train_stimuli is None:
            train_stimuli = train_stimuli_
            test_stimuli = test_stimuli_
        else:
            # Check: stimuli are in the same order across subjects
            assert np.array_equal(train_stimuli, train_stimuli_)
            assert np.array_equal(test_stimuli, test_stimuli_)
        globals()[f'train_responses_subjects'] = train_responses_subjects
        globals()[f'test_responses_subjects'] = test_responses_subjects
        globals()[f'train_stimuli'] = train_stimuli
        globals()[f'test_stimuli'] = test_stimuli

    for split in SPLITS:
        image_paths = globals()[f'{split}_stimuli']
        neural_responses = globals()[f'{split}_responses_subjects']

        stimulus_set = load_stimulus_set(things_image_db_dir, image_paths, split)
        globals()[f'assembly_{split}'] = load_brain_data(neural_responses, stimulus_set, split)
        globals()[f'stimulus_set_{split}'] = stimulus_set
        del stimulus_set
        del neural_responses
    
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
    
    print("Uploading train neural data assembly...")
    assy_train_info = package_data_assembly(
            catalog_identifier=None,  # catalogs are deprecated
            proto_data_assembly=assembly_train, 
            assembly_identifier=assembly_train.name,
            stimulus_set_identifier=stimulus_set_train.name,
            assembly_class_name="NeuroidAssembly",
            bucket_name=bucket_name,
        )
    
    print("Uploading test neural data assembly...")
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
    parser = ArgumentParser(description="Package THINGS fMRI data")
    parser.add_argument(
        '--neural-data-dir', '--neuro-dir',
        dest='neural_data_dir',
        type=str,
        required=True,
        help='Path to the neural data directory containing fMRI .csv and .h5 files.'
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
    
    package_data(neural_data_dir, things_image_db_dir, bucket_name, output_path)
