from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

stimuli = []
image_paths = {}
stimuli_directory = 'images'


'''

Dataset Information:
    1) From Jacob 2020: https://www.nature.com/articles/s41467-021-22078-3
    2) This stimulus set is only six (6) images:
        - occlusion_notched
        - occlusion_adjacent
        - occlusion_occluded
        - depth_notched
        - depth_adjacent
        - depth_occluded
        These six images correspond to Figure 5C in Jacob 2020. The first three are the top row from the figure, right 
        to left in that order. The second three are the bottom row from the figure, right to left, in that order.

        
  
    3) Sample name: occlusion_notched.jpg
    4) Fields:
        - experiment_type: string in the set {"occlusion", "depth"}, indicating the sub-experiment 
        - image_variation - string in the set {"notched", "adjacent", "occluded"} indicating shape variation
         
'''

for filepath in Path(stimuli_directory).glob('*.png'):

    # entire name of image file:
    image_id = filepath.stem

    # parse the image id:
    items = image_id.split("_")

    # get the type of experiment:
    experiment_type = items[0]

    # get the type of image variation:
    image_variation = items[1]

    # combine into stimulus set
    image_paths[image_id] = filepath
    stimuli.append({
        'stimulus_id': image_id,
        'experiment_type': experiment_type,
        'image_variation': image_variation,
    })

# package stimulus_set
stimuli = StimulusSet(stimuli)
stimuli.stimulus_paths = image_paths

# give the StimulusSet an identifier name
stimuli.name = 'Jacob2020_occlusion_depth_ordering'

# upload to S3
package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainio-brainscore")
