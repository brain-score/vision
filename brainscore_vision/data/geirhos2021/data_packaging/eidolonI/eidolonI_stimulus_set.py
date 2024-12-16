from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

stimuli = []
image_paths = {}
stimuli_directory = '../datasets/eidolonI/dnn/session-1'


'''
Dataset Meta Info (from https://github.com/rgeirhos/generalisation-humans-DNNs)

Sample image from dataset:
0001_eid_dnn_1-10-10_oven_10_n04111531_17072.png

This is a concatenation of the following information (separated by '_'):

    1) a four-digit number starting with 0000 for the first image in an experiment; 
       the last image therefore has the number n-1 if n is the number of images in a certain experiment
    2) short code for experiment name, e.g. 'eid' for eidolon-experiment
    3) either e.g. 's01' for 'subject-01', or 'dnn' for DNNs
    4) condition
    5) category (ground truth)
    6) a number (just ignore it)
    7) image identifier in the form a_b.JPEG (or a_b.png), with a being the 
       WNID (WordNet ID) of the corresponding synset and b being an integer.
'''

for filepath in Path(stimuli_directory).glob('*.png'):

    # entire name of image file:
    image_id = filepath.stem
    image_id_long = image_id
    split_name = filepath.stem.split('_')

    # ensure proper metadata length per image in set
    assert len(split_name) == 8

    # Dataset image data, 1-7 from above:
    image_number = split_name[0]
    experiment_code = split_name[1]
    subject = split_name[2]
    condition = split_name[3]
    category_ground_truth = split_name[4]
    random_number = split_name[5]

    # note the split, for a total of 8 metadata fields:
    wordnet_a = split_name[6]
    wordnet_b = split_name[7]

    # image lookup ID, same as data assembly. This is the exact image shown to participant.
    # This is needed, as the raw images have "dnn" in the subject field, even when a human subject
    # was used. Otherwise, the image name data in the raw data table and the image names themselves are the same.
    image_lookup_id = "_".join(split_name[3:])

    image_paths[image_id] = filepath
    stimuli.append({
        'image_id': image_lookup_id,
        'image_id_long': image_id_long,
        'image_number': image_number,
        'experiment_code': experiment_code,
        'condition': condition,
        'truth': category_ground_truth,
        'category_ground_truth': category_ground_truth,
        'random_number': random_number,
        'wordnet_a': wordnet_a,
        'wordnet_b': wordnet_b,
    })

stimuli = StimulusSet(stimuli)
image_id_to_lookup = dict(zip(stimuli['image_id_long'], stimuli['image_id']))
stimuli.image_paths = image_paths
stimuli.image_paths = {image_id_to_lookup[image_id]: path
                       for image_id, path in stimuli.image_paths.items()}
stimuli.name = 'Geirhos2021_eidolonI'  # give the StimulusSet an identifier name

# Ensure 1280 images in dataset
assert len(stimuli) == 1280

# upload to S3
package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainscore-storage/brainio-brainscore")
