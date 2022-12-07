from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

stimuli = []
image_paths = {}
stimuli_directory = '../datasets/edge'


'''
Dataset Meta Info (from https://github.com/rgeirhos/generalisation-humans-DNNs)

** NOTE: This benchmark (edge) has a very different stimulus sets then others!! **

Sample image from dataset:
bear4.png

This is a concatenation of the following information:

    1) Ground Truth category in the 16-set of:
        {airplane, bear, bicycle, bird, boat, bottle, car, cat, 
        chair, clock, dog, elephant, keyboard, knife, oven, truck} 
    2) The sub-image number, in the set {1, 2, 3, ... 10}
    
There are thus 100 total images in this stimulus set (16 categories * 10 variations = 160) 

'''

for filepath in Path(stimuli_directory).glob('*.png'):

    # entire name of image file:
    image_id = filepath.stem

    # re to parse image names into component parts:
    import re
    match = re.match(r"([a-z]+)([0-9]+)", image_id, re.I)
    if match:
        items = match.groups()
    else:
        items = []

    split_name = items

    # ensure proper metadata length per image in set
    assert len(split_name) == 2

    # Dataset image data, 1-7 from above:
    image_category = split_name[0]
    image_variation = split_name[1]

    image_paths[image_id] = filepath
    stimuli.append({
        'image_id': image_id,
        'image_category': image_category,
        'truth': image_category,
        'image_variation': image_variation,
        'condition': 0,
    })

stimuli = StimulusSet(stimuli)
stimuli.image_paths = image_paths
stimuli.name = 'brendel.Geirhos2021_edge'  # give the StimulusSet an identifier name

# Ensure 100 images in dataset
assert len(stimuli) == 160

# upload to S3
package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainio-brainscore")
