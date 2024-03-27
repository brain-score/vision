import csv
from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set


# every stimulus set is separate, incl. baseline condition
STIMULUS_SETS = ['short-2', 'short-4', 'short-6', 'short-8', 'short-16', 'equal-2',
                 'long-2', 'equal-16', 'long-16', 'vernier-only', 'short-2_fit',
                 'short-4_fit', 'short-6_fit', 'short-8_fit', 'short-16_fit',
                 'equal-2_fit', 'long-2_fit', 'equal-16_fit', 'long-16_fit']
DATASET_LENGTHS = {'test': 1225, 'fit': 1225}


def collect_malania_stimulus_set(root_directory, dataset):
    """
    Dataset Meta Info

    Reported in pixels:
        - image_size_x; image_size_y
        - vernier_position_x; vernier_position_y

    Reported in arcsec:
        - vernier_height (height of the vernier elements combined, - middle gap)
        - vernier_offset (horizontal offset between flankers)
        - flanker_height (height of the flanker elements)
        - flanker_spacing (distance between a flanker element and another flanker element)
        - line_width (width of all the lines in all elements)
        - flanker_distance (distance between a flanker and a vernier)
    """
    stimuli = []
    stimulus_paths = {}

    dataset_type = 'fit' if dataset[-3:] == 'fit' else 'test'
    metadata_directory = Path(f'{root_directory}/{dataset}/metadata.csv')
    image_directory = Path(f'{root_directory}/{dataset}/images')
    with open(metadata_directory, 'r') as metadata:
        reader = csv.DictReader(metadata)
        for row in reader:
            stimuli.append({
                'image_size_x': int(row['image_size_x']),
                'image_size_y': int(row['image_size_y']),
                'image_size_c': int(row['image_size_c']),
                'image_size_degrees': float(row['image_size_degrees']),
                'vernier_height': float(row['vernier_height']),
                'vernier_offset': float(row['vernier_offset']),
                'image_label': row['image_label'],
                'flanker_height': float(row['flanker_height']),
                'flanker_spacing': float(row['flanker_spacing']),
                'line_width': float(row['line_width']),
                'flanker_distance': float(row['flanker_distance']),
                'num_flankers': int(row['num_flankers']),
                'vernier_position_x': int(row['vernier_position_x']),
                'vernier_position_y': int(row['vernier_position_y']),
                'stimulus_id': str(row['stimulus_id']),
            })
            stimulus_paths[row['stimulus_id']] = Path(f'{image_directory}/{row["filename"]}')

    stimuli = StimulusSet(stimuli)
    stimuli.stimulus_paths = stimulus_paths
    stimuli.name = f'Malania2007_{dataset}'  # give the StimulusSet an identifier name

    # Ensure expected number of stimuli in datasets
    assert len(stimuli) == DATASET_LENGTHS[dataset_type]
    return stimuli


def return_local_stimulus_set(dataset):
    root_directory = Path(r'../../../packaging/malania2007/malania2007_stimulus_set')
    stimuli = collect_malania_stimulus_set(root_directory, dataset)
    return stimuli


if __name__ == '__main__':
    root_directory = Path(r'../../../packaging/malania2007/malania2007_stimulus_set')
    for stimulus_set in STIMULUS_SETS:
        stimuli = collect_malania_stimulus_set(root_directory, stimulus_set)

        # upload to S3
        #package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
        #                     bucket_name="brainio-brainscore")
