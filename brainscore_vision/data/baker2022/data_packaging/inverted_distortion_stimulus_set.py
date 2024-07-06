from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set
import pandas as pd

stimuli = []
image_paths = {}
stimuli_directory = 'images'


'''
Dataset Information:
- From Baker 2022: https://www.sciencedirect.com/science/article/pii/S2589004222011853#sec9
- 4320 trials total, across 12 subjects. Each subject saw the same 360 images, composed of:
    - 90 inverted-whole, 90 inverted-frankenstein, 90-normal frankenstein, 90-normal whole.
    - of those 4 subtypes, they are further broken into the same 10 images of the set of 9 categories below.

Fields:
1) ground_truth: the base object, in set above
2) image_type:  a string in the set {w, f} for {whole, frankenstein} respectively.
    Note: inverted only hase whole and frankenstein images, no fragmented 
3) image_number: a number {1,2...40} indicates the image variation

'''
categories = ['bear', 'bunny', 'cat', 'elephant', 'frog', 'lizard', 'tiger', 'turtle', 'wolf']
images_actually_shown = pd.read_csv('human_data/inverted.csv')
images_actually_shown = set(images_actually_shown["FileName"].values)

for filepath in Path(stimuli_directory).glob('*.jpg'):

    # entire name of image file:
    image_id = filepath.stem

    if image_id in images_actually_shown:
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
            'image_type_entire': "whole" if image_type is "w" else "frankenstein",
            'image_number': image_number,
            "orientation": "normal" if "inv" not in image_id else "inverted",
        })
    else:
        pass

stimuli = StimulusSet(stimuli)
stimuli.stimulus_paths = image_paths


# give the StimulusSet an identifier name
stimuli.name = 'Baker2022_inverted_distortion'

# upload to S3
package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainio-brainscore")
