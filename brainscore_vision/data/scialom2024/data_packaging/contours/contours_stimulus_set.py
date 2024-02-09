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
                    'category': str(row['category']),
                    'dataset': str(row['representation_mode']),
                    'object_id': int(row['object_id']),
                    'percentage_elements': str(row['percentage_elements']),
                    'stimulus_id': str(row['stimulus_id']),
                })
                stimulus_paths[row['stimulus_id']] = Path(f'{stimuli_directory}/{row["file_name"]}')

    stimuli = StimulusSet(stimuli)
    stimuli.stimulus_paths = stimulus_paths
    stimuli.name = f'Scialom_2024-{dataset}'  # give the StimulusSet an identifier name

    return stimuli


if __name__ == '__main__':
    dataset = 'contours'
    percentage_elements = 'contours'
    stimuli_directory = Path(r'../../dataset')
    metadata_filepath = '../../MetaData_Stimuli_experiment.csv'
    test_stimuli = collect_scialom_stimulus_set(dataset, percentage_elements, stimuli_directory, metadata_filepath)

    # Ensure expected number of stimuli in datasets
    assert len(test_stimuli) == 48
    # upload to S3
    #package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
    #                     bucket_name="brainio-brainscore")
