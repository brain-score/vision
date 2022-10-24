from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

stimuli = []
image_paths = {}
stimuli_directory = '../zhu2019/images'


'''
Dataset Meta Info (from https://arxiv.org/pdf/1905.04598.pdf)

contains 500 images from 5 classes, 100/class in the set {aeroplane, bicycle, bus, car, motorbike}

Sample image from dataset:
bicycle_hard_image_87.png

This is a concatenation of the following information (separated by '_'):

    1) ground truth category from image set above
    2) the word hard - not clarified but assumption is the level of occlusion
    3) the word "image"
    4) image number, in the set {1,2,...100} inclusive. 
'''

for filepath in Path(stimuli_directory).glob('*.png'):

    # entire name of image file:
    image_id = filepath.stem
    image_id_long = image_id
    split_name = filepath.stem.split('_')

    # ensure proper metadata length per image in set
    assert len(split_name) == 4

    # Dataset image data, 1-7 from above:
    ground_truth = split_name[0]
    occlusion_strength = split_name[1]
    word_image = split_name[2]
    image_number = split_name[3]

    image_paths[image_id] = filepath
    stimuli.append({
        'stimulus_id': image_id,
        'ground_truth': ground_truth,
        'image_label': ground_truth,
        'occlusion_strength': occlusion_strength,
        'word_image': word_image,
        'image_number': image_number,
    })

stimuli = StimulusSet(stimuli)
stimuli.stimulus_paths = image_paths
stimuli.name = 'yuille.Zhu2019_extreme_occlusion'  # give the StimulusSet an identifier name

# upload to S3
package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainio-brainscore")
