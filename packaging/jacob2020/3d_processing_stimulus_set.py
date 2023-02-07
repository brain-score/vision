from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

stimuli = []
image_paths = {}
stimuli_directory = 'images'


'''
Dataset Information:
- From Jacob 2020: https://www.nature.com/articles/s41467-021-22078-3

- This stimulus set is only six (6) images - two cubes, 2 "y"s, and 2 squares.
- Naming Convention - sample name: cube_1.jpg
    

Fields:
1) shape: string in the set {"cube", "y", "square"}
2) number - int in the set {1, 2} indicating shape variation 
'''

for filepath in Path(stimuli_directory).glob('*.png'):

    # entire name of image file:
    image_id = filepath.stem

    # parse the image id:
    items = image_id.split("_")

    # get image_shape
    image_shape = items[0]

    # get image_number:
    image_number = items[1]

    # combine into stimulus set
    image_paths[image_id] = filepath
    stimuli.append({
        'stimulus_id': image_id,
        'image_shape': image_shape,
        'image_number': image_number,
    })

# package stimulus_set
stimuli = StimulusSet(stimuli)
stimuli.stimulus_paths = image_paths

# give the StimulusSet an identifier name
stimuli.name = 'Jacob2020_3dpi'

# upload to S3
package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainio-brainscore")
