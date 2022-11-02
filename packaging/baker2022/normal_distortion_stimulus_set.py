from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set
import numpy as np

stimuli = []
image_paths = {}
stimuli_directory = 'images'


'''
Dataset Information:

- From Baker 2022: https://www.sciencedirect.com/science/article/pii/S2589004222011853#sec9
- 1800 images total: 1080 normal distortion (** this stimulus set **), 720 inverted distortion
- normal/inverted distortion -> contains the three classes of distortion:
    1) normal image
    2) fragmented image
    3) frankenstein image

- For the normal distortion, there are 40 images/condition, with 9 categories: 
    categories are: {bear, bunny, cat, elephant, frog, lizard, tiger, turtle, wolf}
- For inverted distortion, there are a variable number of images/condition, but 720 total unique images

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

    if "inv" in ground_truth:
        ground_truth = ground_truth.replace("inv", "")
    elif "nv" in ground_truth:
        ground_truth = ground_truth.replace("nv", "")

    image_paths[image_id] = filepath
    stimuli.append({
        'stimulus_id': image_id,
        'animal': ground_truth,
        'image_type': "w" if image_type is "i" else image_type,
        'image_number': image_number,
        "orientation": "normal" if "inv" not in image_id else "inverted",
    })

stimuli = StimulusSet(stimuli)

stimuli.stimulus_paths = image_paths

# remove all inverted stimuli
stimuli = stimuli[stimuli["orientation"] == "normal"]
stimuli.name = 'kellmen.Baker2022_normal_distortion'  # give the StimulusSet an identifier name

# upload to S3
package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainio-brainscore")
