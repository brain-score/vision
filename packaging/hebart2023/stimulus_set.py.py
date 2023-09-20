from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set
import pandas as pd
import re

stimuli = []
image_paths = {}
stimuli_directory = 'images'


'''
Dataset Information:

- From Hebart 2023: link TODO
- 1854 images total

Fields:

- TODO

'''
categories = ['aardvark'] # TODO

for filepath in Path(stimuli_directory).glob('*.jpg'):

    # entire name of image file:
    image_id = filepath.stem
    image_paths[image_id] = filepath
    stimuli.append({
        'stimulus_id': image_id,
        'label': image_id[:-4],
        'number': None # TODO this might 
    })

stimuli = StimulusSet(stimuli)

stimuli.stimulus_paths = image_paths

# upload to S3
package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainio-brainscore")