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
            'segments-21', 'segments-27', 'segments-35', 'segments-46', 'segments-59', 'segments-77', 'segments-100',
            'phosphenes-composite', 'segments-composite']
PERCENTAGE_ELEMENTS = {'rgb': 'RGB', 'contours': 'contours', 'phosphenes-12': 12, 'phosphenes-16': 16,
                       'phosphenes-21': 21, 'phosphenes-27': 27, 'phosphenes-35': 35, 'phosphenes-46': 46,
                       'phosphenes-59': 59, 'phosphenes-77': 77, 'phosphenes-100': 100, 'segments-12': 12,
                       'segments-16': 16, 'segments-21': 21, 'segments-27': 27, 'segments-35': 35, 'segments-46': 46,
                       'segments-59': 59, 'segments-77': 77, 'segments-100': 100, 'phosphenes-composite': 'all',
                       'segments-composite': 'all'}


def collect_scialom_stimulus_set(dataset, percentage_elements, stimuli_directory, metadata_filepath, which_composite):
    stimuli = []
    stimulus_paths = {}
    dataset = dataset.split('-')[0]

    with open(metadata_filepath, 'r') as metadata:
        reader = csv.DictReader(metadata)
        for row in reader:
            stimulus_meta = {
                'image_height': int(row['image_height']),
                'image_width': int(row['image_width']),
                'num_channels': int(row['channel']),
                'dataset': str(row['representation_mode']),
                'object_id': int(row['object_id']),
                'stimulus_id': str(row['stimulus_id']),
                'truth': str(row['category']),
                'percentage_elements': str(row['percentage_elements']),
            }
            if which_composite is not None:
                if row['representation_mode'].lower() == which_composite or \
                    row['representation_mode'].lower() == 'rgb' or \
                    row['representation_mode'].lower() == 'contours':
                    stimulus_meta = {**stimulus_meta,
                                     'condition': which_composite}
                    stimuli.append(stimulus_meta)
                    stimulus_paths[str(row['stimulus_id'])] = Path(f'{stimuli_directory}/{row["file_name"]}')
            elif row['percentage_elements'] == str(percentage_elements) and \
                    row['representation_mode'].lower() == dataset:
                stimulus_meta = {**stimulus_meta,
                                 'condition': str(row['percentage_elements'])}
                stimuli.append(stimulus_meta)
                stimulus_paths[str(row['stimulus_id'])] = Path(f'{stimuli_directory}/{row["file_name"]}')

    stimuli = StimulusSet(stimuli)
    stimuli.stimulus_paths = stimulus_paths
    if dataset in ['phosphenes', 'segments']:
        stimuli.name = f'Scialom2024_{dataset}-{percentage_elements}'
        stimuli.identifier = f'Scialom2024_{dataset}-{percentage_elements}'
    else:
        stimuli.name = f'Scialom2024_{dataset}'
        stimuli.identifier = f'Scialom2024_{dataset}'

    return stimuli


if __name__ == '__main__':
    stimuli_directory = Path(r'../dataset')
    metadata_filepath = Path('../MetaData_Stimuli_experiment.csv')
    for dataset in DATASETS:
        if dataset == 'phosphenes-composite':
            which_composite = 'phosphenes'
        elif dataset == 'segments-composite':
            which_composite = 'segments'
        else:
            which_composite = None
        percentage_elements = PERCENTAGE_ELEMENTS[dataset]
        stimuli = collect_scialom_stimulus_set(dataset, percentage_elements, stimuli_directory, metadata_filepath,
                                                    which_composite=which_composite)

        # Ensure expected number of stimuli in datasets
        if which_composite is None:
            assert len(stimuli) == 48
        else:
            assert len(stimuli) == 528
        # upload to S3
        package_stimulus_set(catalog_name="brainio_brainscore",
                             proto_stimulus_set=stimuli,
                             stimulus_set_identifier=stimuli.name,
                             bucket_name="brainio-brainscore")