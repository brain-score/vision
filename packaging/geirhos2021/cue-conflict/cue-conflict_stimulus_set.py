from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

stimuli = []
image_paths = {}
stimuli_directory = '../datasets/cue-conflict'


'''
*** NOTE: This benchmark (cue-conflict) has a different schema then all other 16 in 
          the Geirhos set!
***          

Dataset Meta Info (from https://github.com/rgeirhos/generalisation-humans-DNNs)

Sample image from dataset:
airplane1-bicycle2.png

This is a concatenation of the following information:

    1) Ground Truth category image in the 16-set of:
        {airplane, bear, bicycle, bird, boat, bottle, car, cat, 
        chair, clock, dog, elephant, keyboard, knife, oven, truck} 
    2) The sub-image number, in the set {1, 2, 3, ... 10}
    3) Conflict category image in the 16-set of:
        {airplane, bear, bicycle, bird, boat, bottle, car, cat, 
        chair, clock, dog, elephant, keyboard, knife, oven, truck} 
    4) The sub-image number, in the set {1, 2, 3, ... 10}
    
There are 1280 total images in this stimulus set 

'''

for filepath in Path(stimuli_directory).glob('*.png'):

    # entire name of image file:
    image_id = filepath.stem
    original_image = image_id.split("-")[0]
    conflict_image = image_id.split("-")[1]

    # re to parse image names into component parts:
    import re
    match = re.match(r"([a-z]+)([0-9]+)-([a-z]+)([0-9]+)", image_id, re.I)
    if match:
        items = match.groups()
    else:
        items = []

    split_name = items

    # ensure proper metadata length per image in set
    assert len(split_name) == 4

    # Dataset image data, 1-7 from above:
    original_image_category = split_name[0]
    original_image_variation = split_name[1]
    conflict_image_category = split_name[2]
    conflict_image_variation = split_name[3]

    image_paths[image_id] = filepath
    stimuli.append({
        'image_id': image_id,
        'original_image': original_image,
        'conflict_image': conflict_image,
        'original_image_category': original_image_category,
        'original_image_variation': original_image_variation,
        'conflict_image_category': conflict_image_category,
        'conflict_image_variation': conflict_image_variation,

    })

stimuli = StimulusSet(stimuli)
stimuli.image_paths = image_paths
stimuli.name = 'brendel.Geirhos2021_cue-conflict'  # give the StimulusSet an identifier name

# Ensure 100 images in dataset
assert len(stimuli) == 1280

# upload to S3
package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainio-brainscore")
