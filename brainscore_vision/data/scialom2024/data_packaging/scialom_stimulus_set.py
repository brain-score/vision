from pathlib import Path
import csv

from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set


'''
Dataset Meta Info

Reported in pixels:
    - image_height
    - image_width

Others:
    - num_channels (3 in the case of RGB and contours, 1 otherwise)
    - dataset (RGB, contours, phosphenes or segments stimuli)
    - object_id (a unique identifier of the specific exemplar of a given category. For each object category, there are
        4 objects)
    - category (the category of the object; one of:
        'Banana', 
        'Beanie', 
        'Binoculars', 
        'Boot', 
        'Bowl', 
        'Cup', 
        'Glasses',
        'Lamp', 
        'Pan', 
        'Sewing machine', 
        'Shovel', 
        'Truck'
        )
'''

DATASETS = ['rgb', 'contours', 'phosphenes-12', 'phosphenes-16', 'phosphenes-21', 'phosphenes-27', 'phosphenes-35',
            'phosphenes-46', 'phosphenes-59', 'phosphenes-77', 'phosphenes-100', 'segments-12', 'segments-16',
            'segments-21', 'segments-27', 'segments-35', 'segments-46', 'segments-59', 'segments-77', 'segments-100']
PERCENTAGE_ELEMENTS = {'rgb': 'RGB', 'contours': 'contours', 'phosphenes-12': 12, 'phosphenes-16': 16,
                       'phosphenes-21': 21, 'phosphenes-27': 27, 'phosphenes-35': 35, 'phosphenes-46': 46,
                       'phosphenes-59': 59, 'phosphenes-77': 77, 'phosphenes-100': 100, 'segments-12': 12,
                       'segments-16': 16, 'segments-21': 21, 'segments-27': 27, 'segments-35': 35, 'segments-46': 46,
                       'segments-59': 59, 'segments-77': 77, 'segments-100': 100}


def collect_scialom_stimulus_set(dataset, percentage_elements, stimuli_directory, metadata_filepath):
    stimuli = []
    stimulus_paths = {}

    with open(metadata_filepath, 'r') as metadata:
        reader = csv.DictReader(metadata)
        for row in reader:
            if row['percentage_elements'] == percentage_elements and row['representation_mode'] == dataset:
                stimuli.append({
                    'image_height': int(row['image_height']),
                    'image_width': int(row['image_width']),
                    'num_channels': int(row['channel']),
                    'dataset': str(row['representation_mode']),
                    'object_id': int(row['object_id']),
                    'stimulus_id': int(row['stimulus_id']),
                    'condition': str(row['percentage_elements']),
                    'truth': str(row['category'])
                })
                # needed changing
                stimulus_paths[int(row['stimulus_id'])] = Path(f'{stimuli_directory}/{row["file_name"]}')

    stimuli = StimulusSet(stimuli)
    stimuli.stimulus_paths = stimulus_paths
    if dataset in ['phosphenes', 'segments']:
        stimuli.name = f'Scialom2024_{dataset}-{percentage_elements}'
        stimuli.identifier = f'Scialom_2024-{dataset}-{percentage_elements}'
    else:
        stimuli.name = f'Scialom2024_{dataset}'
        stimuli.identifier = f'Scialom_2024-{dataset}'

    return stimuli


if __name__ == '__main__':
    stimuli_directory = Path(r'../../dataset')
    metadata_filepath = '../../MetaData_Stimuli_experiment.csv'
    for dataset in DATASETS:
        percentage_elements = PERCENTAGE_ELEMENTS[dataset]
        test_stimuli = collect_scialom_stimulus_set(dataset, percentage_elements, stimuli_directory, metadata_filepath)

        # Ensure expected number of stimuli in datasets
        assert len(test_stimuli) == 48
        # upload to S3
        #package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
        #                     bucket_name="brainio-brainscore")