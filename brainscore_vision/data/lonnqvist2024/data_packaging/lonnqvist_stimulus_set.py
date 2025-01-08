from pathlib import Path
import csv

from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

'''
dataset Meta Info

- curve length: values from 20-200 in steps of 10 
- n_cross: number of times the lines intercept, range from 1-7
- condition: same or diff
'''


def collect_lonnqvist_stimulus_set(dataset, stimuli_directory, metadata_filepath):
    stimuli = []
    stimulus_paths = {}

    with open(metadata_filepath, 'r') as metadata:
        reader = csv.DictReader(metadata)
        for row in reader:
            stimulus_meta = {
                'curve_length': int(row['curve_length']),
                'n_cross': int(row['n_cross']),
                'image_path': str(row['path']),
                'stimulus_id': str(row['idx']),
                'truth': str(row['correct_response_key']),
                'image_label': str(row['correct_response_key'])
            }

            stimuli.append(stimulus_meta)
            stimulus_paths[str(row['idx'])] = Path(f'{row["path"]}')

    stimuli = StimulusSet(stimuli)
    stimuli.stimulus_paths = stimulus_paths

    stimuli.name = f'Lonnqvist2024_{dataset}'
    stimuli.identifier = f'Lonnqvist2024_{dataset}'
    return stimuli


if __name__ == '__main__':
    datasets = ['train', 'test']
    stimulus_directories = {'train': Path(r'stimuli/images_examples'),
                           'test': Path(r'stimuli/images')}
    metadata_filepaths = {'train': Path('stimuli/metadata_examples.csv'),
                            'test': Path('stimuli/metadata.csv')}
    for dataset in datasets:
        stimulus_set = collect_lonnqvist_stimulus_set(dataset,
                                                      stimulus_directories[dataset],
                                                      metadata_filepaths[dataset])
        if dataset == 'train':
            assert len(stimulus_set) == 185
        else:
            assert len(stimulus_set) == 380
        prints = package_stimulus_set(catalog_name=None,
                                      proto_stimulus_set=stimulus_set,
                                      stimulus_set_identifier=stimulus_set.name,
                                      bucket_name="brainscore-storage/brainio-brainscore")
        print(prints)
