from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set
import os
import pandas as pd
from pathlib import Path
import numpy as np

SUBJECTS = ["subject1", "subject2", "subject3", "subject4", "subject5", "subject6", "subject7", "subject8"]

def collect_nsd_stimulus_set(root_directory, subject):
    """
    Dataset Meta Info (from https://naturalscenesdataset.org/ ; https://cocodataset.org/) 

    The Natural Scenes Dataset (NSD) is a large-scale fMRI dataset conducted at ultra-high-field (7T) strength 
    at the Center of Magnetic Resonance Research (CMRR) at the University of Minnesota. The dataset consists of 
    whole-brain, high-resolution (1.8-mm isotropic, 1.6-s sampling rate) fMRI measurements of 8 healthy adult 
    subjects while they viewed thousands of color natural scenes over the course of 30–40 scan sessions. 
    While viewing these images, subjects were engaged in a continuous recognition task in which they reported 
    whether they had seen each given image at any point in the experiment. These data constitute a massive 
    benchmark dataset for computational models of visual representation and cognition, and can support a 
    wide range of scientific inquiry.

    """

    metadata_directory = Path(f'{root_directory}/metadata/nsd_stim_info_merged.csv')
    image_directory = Path(f'{root_directory}/nsd_stimulus_set')
    subject_directory = Path(f'{root_directory}/subj-data/subj0{subject[-1]}')
    
    s_info = pd.read_csv(metadata_directory, index_col=0)

    stimulus_id_path = sorted(os.listdir(image_directory))
    filename = [i for i in stimulus_id_path]
    stimulus_id = [i.rsplit('_', 1)[1][:-4] for i in stimulus_id_path]
    subject_columns = SUBJECTS
    s_info["subject"] = s_info[subject_columns].idxmax(axis=1)
    s_info.loc[s_info[subject_columns].sum(axis=1) == len(subject_columns), "subject"] = "shared"
    s_info['stimulus_id'] = stimulus_id
    s_info['filename'] = filename
    s_info['nsdId'] = s_info['nsdId'].apply(lambda x: format(x, "05"))

    # s_info.drop(subject_columns + [f'{subj}_rep{i}' for subj in subject_columns for i in range(3)], axis=1,  inplace=True)
    s_info.drop(['subject1', 'subject2', 'subject3',
        'subject4', 'subject5', 'subject6', 'subject7', 'subject8'
        , 'subject1_rep0', 'subject1_rep1', 'subject1_rep2', 'subject2_rep0',
        'subject2_rep1', 'subject2_rep2', 'subject3_rep0', 'subject3_rep1',
        'subject3_rep2', 'subject4_rep0', 'subject4_rep1', 'subject4_rep2',
        'subject5_rep0', 'subject5_rep1', 'subject5_rep2', 'subject6_rep0',
        'subject6_rep1', 'subject6_rep2', 'subject7_rep0', 'subject7_rep1',
        'subject7_rep2', 'subject8_rep0', 'subject8_rep1', 'subject8_rep2'
        ], axis=1,  inplace = True)
    
    s_info['nsd_ids'] = range(0, 73000)

    s_info = s_info[s_info.subject.isin([subject, 'shared'])]

    # #add the object category and supercategory 
    # coco_metadata = pd.read_csv(COCO_META_DATA_PATH)
    # s_info['category'] = coco_metadata.category[coco_metadata.cocoId.isin(s_info.cocoId)]
    # s_info['supercategory'] = coco_metadata.supercategory[coco_metadata.cocoId.isin(s_info.cocoId)]
    # s_info['object_name'] = s_info.category

    # add a column representing the train and test condition in the algonauts challenge for the chosen subject 
    # Step 1: Read the contents of train and test CSV files into lists
    train_csv_path = Path(f'{subject_directory}/training_split/imgs-lookup-train.csv')
    test_csv_path = Path(f'{subject_directory}/test_split/imgs-lookup-test.csv')

    # Step 2: Load the CSV files and extract the last 46 characters
    train_lookup = np.squeeze(pd.read_csv(train_csv_path, header=None)).str[-46:].tolist()
    test_lookup = np.squeeze(pd.read_csv(test_csv_path, header=None)).str[-46:].tolist()

    # Step 3: Iterate through rows and add "train/test" column
    train_test_column = []
    for _, row in s_info.iterrows():
        filename = row['filename']
        if filename in train_lookup:
            train_test_column.append('train')
        elif filename in test_lookup:
            train_test_column.append('test')
        else:
            train_test_column.append('unknown')  # Handle cases not found in either set

    s_info['algonauts_train_test'] = train_test_column

    s_info = s_info[s_info['algonauts_train_test'] != 'unknown']

    # Step 3: Sort the DataFrame based on "train_test" and "nsd_ids" columns
    s_info.sort_values(by=['algonauts_train_test', 'nsd_ids'], ascending=[False, True],  inplace=True)

    stimuli = StimulusSet(s_info)

    stimuli.stimulus_paths = {row['stimulus_id']: Path(image_directory) / row['filename'] for _, row in stimuli.iterrows()}

    # Remove the 'filename' column from the StimulusSet DataFrame
    stimuli.drop('filename', axis=1, inplace=True)
    
    stimuli.name = f'NSD2022_{subject}_stimulus_set'
   
    return stimuli


if __name__ == '__main__':

    root_directory = Path(r'./')

    for subject in SUBJECTS:
        stimuli = collect_nsd_stimulus_set(root_directory, subject)
        # upload to S3
        print(stimuli.name)
        package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                            bucket_name="brainio-brainscore")
    



