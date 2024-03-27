import csv
from pathlib import Path

from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

'''
Dataset Meta Info

Reported in pixels:
    - image_size_x_pix; image_size_y_pix
    - vernier_position_x_pix; vernier_position_y_pix

Reported in arcsec:
    - vernier_height_arcsec (height of the vernier elements combined, - middle gap)
    - vernier_offset_arcsec (horizontal offset between flankers)
    - flanker_height_arcsec (height of the flanker elements)
    - flanker_spacing_arcsec (distance between a flanker element and another flanker element)
    - line_width_arcsec (width of all the lines in all elements)
    - flanker_distance_arcsec (distance between a flanker and a vernier)
'''


def collect_malania_stimulus_set(root_directory, dataset):
    stimuli = []
    stimulus_paths = {}

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

    return stimuli


if __name__ == '__main__':
    stimulus_root_directory = Path(r'../stimuli')
    test_stimuli = collect_malania_stimulus_set(stimulus_root_directory, 'short-16')

    # Ensure expected number of stimuli in datasets
    assert len(test_stimuli) == 50

    # upload to S3
    #package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
    #                     bucket_name="brainio-brainscore")

    train_stimuli = collect_malania_stimulus_set(stimulus_root_directory, 'short-16_fit')

    # Ensure expected number of stimuli in datasets
    assert len(train_stimuli) == 500

    # upload to S3
    # package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
    #                     bucket_name="brainio-brainscore")
