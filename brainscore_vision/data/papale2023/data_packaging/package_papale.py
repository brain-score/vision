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

#here importing mat73 is fine:
import mat73

DEFAULT_BUCKET_NAME = "brainscore-storage/brainio-brainscore" 
DEFAULT_OUTPUT_PATH = "./uploaded_data_info/"

SUBJECTS = ['monkeyF', 'monkeyN']
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


MIN_RELIABILITY = 0.3 #electrodes that do not pass this threshold will be excluded


def package_everything(neural_data_dir: Path, things_image_db_dir: Path, bucket_name: str, output_path: Path) -> None:
    """
    Args:
        neural_data_dir: Path to the neural data directory containing .mat files
        things_image_db_dir: Path to the stimulus dir containing images, should normally end in /object_images/
        bucket_name: S3 bucket name for upload
        output_path: Directory path where file hashes will be saved
    """
    # use subject 1's metadata as reference
    metadata_subj1 = mat73.loadmat(neural_data_dir / SUBJECTS[0] / '_logs' / 'things_imgs.mat')


    print(f" each key ['train_imgs', 'test_imgs'] in the metadata has subkeys {metadata_subj1['train_imgs'].keys()}")
    for subkey in metadata_subj1['train_imgs'].keys():
        print(f'subkey: {subkey} is a list with data like: {metadata_subj1["train_imgs"][subkey][0]}')

    for subject in SUBJECTS[1:]:
            subject_metadata = mat73.loadmat(neural_data_dir / subject / '_logs' / 'things_imgs.mat')
            for split in ['train_imgs', 'test_imgs']:
                for key in ['class', 'things_path']:
                    assert metadata_subj1[split][key] == subject_metadata[split][key], f"Mismatch in {split} for key: {key}"
    print("Metadata is consistent across subjects.")

    #check that the images really are at the given path
    for key in ['train_imgs', 'test_imgs']:
        check_path_str = metadata_subj1[key]['things_path'][0].replace('\\', '/')
        check_path = things_image_db_dir / check_path_str
        assert check_path.exists(), f"No image at: {check_path}, please correct the image path"

    # get the stimuli to populate each StimulusSet
    for split in SPLITS:
        stimulus_ids, stimulus_paths, labels = [], {}, {}

        img_classes = metadata_subj1[f'{split}_imgs']['class']
        img_paths = metadata_subj1[f'{split}_imgs']['things_path']
        for img_class, img_path in tqdm(zip(img_classes, img_paths), total=len(img_classes)):
            
            img_path = img_path.replace('\\', '/')
            stimulus_path = things_image_db_dir / img_path
            stimulus_id =  stimulus_path.stem.split('/')[-1]
            
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
        stimulus_set.name = f'THINGS_TVSD_{split}_Stimuli'
        stimulus_set.identifier = f'things_tvsd_{split}_stimuli'

        globals()[f'stimulus_set_{split}'] = stimulus_set
        print(f'Created StimulusSet for {split} with {len(stimulus_set)} stimuli.')
        del stimulus_set

    # neural data to populate each NeuroidAssembly
    for split in SPLITS:
        whole_split_neural = defaultdict(dict)
        for subject in SUBJECTS:
            print(f'Processing {split} split of subject {subject}')
            file_path = neural_data_dir / subject / 'THINGS_normMUA.mat'
            neural_data = mat73.loadmat(file_path)
            neural_responses = neural_data[f'{split}_MUA']
            reliability = neural_data['reliab']
            
            electrode_reliability = reliability.mean(axis=1)
            is_reliable = electrode_reliability > MIN_RELIABILITY
            assert neural_responses.shape[0] == is_reliable.shape[0], "Mismatch in number of electrodes"

            for region, (start, end) in REGION_CHANNELS_MAP[subject].items(): 
                neuroid_start, neuroid_end = start, end+1
                neural_responses_region = neural_responses[neuroid_start:neuroid_end].T
                assert neural_responses_region.shape[1] == (neuroid_end - neuroid_start), "Mismatch in region electrode count"
                neural_responses_region = neural_responses_region[:, is_reliable[neuroid_start:neuroid_end]]
                whole_split_neural[subject][region] = neural_responses_region
                print(f"in {region} for {subject} there were {neural_responses_region.shape[1]} reliable electrodes "+\
                    f"out of {(neuroid_end - neuroid_start)} total electrodes " +\
                    f'final shape: {neural_responses_region.shape}')

        data_neural_concat = [] #to flatten the whole_split_neural[subject][region] into neuroid x trial x time_bin
        subject_indices, region_indices, neuroid_indices = [], [], []
        for subject in SUBJECTS:
            total_neuroid_per_subject = 0
            for region in REGIONS:
                neural_responses_region = whole_split_neural[subject][region]
                data_neural_concat.append(neural_responses_region)
                
                n_neuroids = neural_responses_region.shape[1]

                subject_indices.extend([subject] * n_neuroids)
                region_indices.extend([region] * n_neuroids)
                total_neuroid_per_subject += n_neuroids
            
            neuroid_indices.extend(range(total_neuroid_per_subject))

        data_neural_concat = np.concatenate(data_neural_concat, axis=1)
        data_neural_flatten = data_neural_concat.reshape(data_neural_concat.shape[0], -1)

        stimulus_set = globals()[f'stimulus_set_{split}']
        assembly = NeuroidAssembly(
            data_neural_flatten.reshape(
                data_neural_concat.shape[0], data_neural_concat.shape[1], 1),
            dims=["presentation", "neuroid", "time_bin"],
            coords={
                "stimulus_id": ("presentation", stimulus_set["stimulus_id"].values),
                "stimulus_label": ("presentation", stimulus_set["label"].values),
                "object_name": ("presentation", stimulus_set["object_name"].values),
                "stimulus_label_idx": ("presentation", stimulus_set["label_idx"].values),
                "region": ("neuroid", region_indices),
                "subject": ("neuroid", subject_indices),
                "neuroid_id": ("neuroid", neuroid_indices),
                "time_bin_start": ("time_bin", [70]),
                "time_bin_end": ("time_bin", [170])
            },
        )

        # Assign a name to the data assembly
        assembly.name = f'THINGS_TVSD_{split}_Assembly'
        assembly.attrs['stimulus_set'] = stimulus_set


        # Ensure the assembly's stimulus IDs match those provided in the stimulus set
        assert np.array_equal(
            assembly["stimulus_id"].values, stimulus_set["stimulus_id"].values
        ), "Stimulus IDs do not match."
        
        globals()[f'assembly_{split}'] = assembly
        del stimulus_set
        del assembly

    stimulus_set_train = globals()['stimulus_set_train']
    stimulus_set_test = globals()['stimulus_set_test']
    assembly_train = globals()['assembly_train']
    assembly_test = globals()['assembly_test']

    print("Branio is now packaging the stimuli for upload. Don't worry, this might take a moment")
    print("Uploading train stimuli...")
    stim_train_info = package_stimulus_set(
                            catalog_name=None, #catalogs are deprecated
                            proto_stimulus_set=stimulus_set_train,
                            stimulus_set_identifier=stimulus_set_train.name,
                            bucket_name=bucket_name)
    print("Uploading test stimuli...")
    stim_test_info = package_stimulus_set(
                            catalog_name=None, #catalogs are deprecated
                            proto_stimulus_set=stimulus_set_test,
                            stimulus_set_identifier=stimulus_set_test.name,
                            bucket_name=bucket_name)
    print('Hashes and ids of StimulusSets:')
    print(stim_train_info)
    print(stim_test_info)

    print("Uploading train neural data assembly...")
    assy_train_info = package_data_assembly(
            catalog_identifier=None, #catalogs are deprecated
            proto_data_assembly=assembly_train, 
            assembly_identifier=assembly_train.name,
            stimulus_set_identifier=stimulus_set_train.name,
            assembly_class_name="NeuroidAssembly",
            bucket_name=bucket_name,
        )

    print("Uploading test neural data assembly...")
    assy_test_info = package_data_assembly(
            catalog_identifier=None, #catalogs are deprecated
            proto_data_assembly=assembly_test, 
            assembly_identifier=assembly_test.name,
            stimulus_set_identifier=stimulus_set_test.name,
            assembly_class_name="NeuroidAssembly",
            bucket_name=bucket_name,
        )
    print('Hashes and ids of Assemblies:')
    print(assy_train_info)
    print(assy_test_info)

    #save the info jsons to the file system
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
    parser = ArgumentParser(description="Package TVSD data")
    parser.add_argument(
        '--neural-data-dir', '--neuro-dir', 
        dest='neural_data_dir', 
        type=str, 
        required=True,
        help='Path to the neural data directory containing .mat files.'
    )
    parser.add_argument(
        '--things-image-db-dir', '--imgs-dir', 
        dest='things_image_db_dir', 
        type=str, 
        required=True,
        help='Path to the stimulus data directory containing images.'
    )
    parser.add_argument( #optional bucket
        '--bucket-name',
        dest='bucket_name',
        type=str,
        default=DEFAULT_BUCKET_NAME,
        help='the adress of the S3 bucket, where the files are uploaded.'
    )
    parser.add_argument(
        '--output-path',
        dest='output_path',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help='where the hashes of the files are saved.'
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
