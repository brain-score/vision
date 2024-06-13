import csv
from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set


# every stimulus set is separate, incl. baseline condition
STIMULUS_SETS = ['short-2', 'short-4', 'short-6', 'short-8', 'short-16', 'equal-2',
                 'long-2', 'equal-16', 'long-16', 'vernier-only', 'short-2_fit',
                 'short-4_fit', 'short-6_fit', 'short-8_fit', 'short-16_fit',
                 'equal-2_fit', 'long-2_fit', 'equal-16_fit', 'long-16_fit']
DATASET_LENGTHS = {'test': 50, 'fit': 500}


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
                'image_size_x_pix': int(row['image_size_x_pix']),
                'image_size_y_pix': int(row['image_size_y_pix']),
                'image_size_c': int(row['image_size_c']),
                'image_size_degrees': float(row['image_size_degrees']),
                'vernier_height_arcsec': float(row['vernier_height_arcsec']),
                'vernier_offset_arcsec': float(row['vernier_offset_arcsec']),
                'image_label': row['image_label'],
                'flanker_height_arcsec': float(row['flanker_height_arcsec']),
                'flanker_spacing_arcsec': float(row['flanker_spacing_arcsec']),
                'line_width_arcsec': float(row['line_width_arcsec']),
                'flanker_distance_arcsec': float(row['flanker_distance_arcsec']),
                'num_flankers': int(row['num_flankers']),
                'vernier_position_x_pix': int(row['vernier_position_x_pix']),
                'vernier_position_y_pix': int(row['vernier_position_y_pix']),
                'stimulus_id': str(row['stimulus_id']),
            })
            stimulus_paths[row['stimulus_id']] = Path(f'{image_directory}/{row["filename"]}')

    stimuli = StimulusSet(stimuli)
    stimuli.stimulus_paths = stimulus_paths
    stimuli.name = f'Malania2007_{dataset}'  # give the StimulusSet an identifier name
    stimuli.identifier = f'Malania2007_{dataset}'

    # Ensure expected number of stimuli in datasets
    assert len(stimuli) == DATASET_LENGTHS[dataset_type]
    return stimuli


if __name__ == '__main__':
    root_directory = Path(r'../../data/malania2007/data_packaging/')
    for stimulus_set in STIMULUS_SETS:
        stimuli = collect_malania_stimulus_set(root_directory, stimulus_set)

        # upload to S3
        prints = package_stimulus_set(catalog_name=None,
                                      proto_stimulus_set=stimuli,
                                      stimulus_set_identifier=stimuli.name,
                                      bucket_name="brainio-brainscore")
        print(prints)
