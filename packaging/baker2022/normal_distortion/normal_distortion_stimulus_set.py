from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

stimuli = []
image_paths = {}
stimuli_directory = '../images'


'''
Dataset Information:

- From Baker 2022: https://www.sciencedirect.com/science/article/pii/S2589004222011853#sec9
- 120 * 9 = 1080 images
- normal distortion -> contains the three classes of distortion:
    1) normal image
    2) fragmented image
    3) frankenstein image

- There are 40 images/condition, with 9 categories. 
- categories are: {bear, bunny, cat, elephant, frog, lizard, tiger, turtle, wolf}

Fields:

1) ground_truth: the base object, in set above
2) image_type:  a string in the set {w, f, o} for {whole, frankenstein, fragmented} respectively. 
3) image_number: a number {1,2...40} indicates the image variation

'''
categories = ['bear', 'bunny', 'cat', 'elephant', 'frog', 'lizard', 'tiger', 'turtle', 'wolf']

for filepath in Path(stimuli_directory).glob('*.jpg'):

    # entire name of image file:
    image_id = filepath.stem

    import re
    match = re.match(r"([a-z]+)([0-9]+)", image_id, re.I)
    if match:
        items = match.groups()
    else:
        items = ["", ""]

    # ground truth
    ground_truth = items[0]

    # image_number:
    image_number = items[1]

    # parse the needed image type letter
    if ground_truth in categories:
        image_type = "w"
    else:
        image_type = ground_truth[0]
        ground_truth = ground_truth[1:]

    image_paths[image_id] = filepath
    stimuli.append({
        'image_id': image_id,
        'ground_truth': ground_truth,
        'image_type': image_type,
        'image_number': image_number,
    })

stimuli = StimulusSet(stimuli)
stimuli.image_paths = image_paths
stimuli.name = 'kellmen.Baker2022_local_configural'  # give the StimulusSet an identifier name

# Ensure 1080 images in dataset
assert len(stimuli) == 1080

# upload to S3
# package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
#                      bucket_name="brainio-brainscore")
