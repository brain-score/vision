from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

stimuli = []
image_paths = {}
stimuli_directory = '../global-distortion-images'


'''
Dataset Information:

- From Baker 2018: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006613
- 6 images
- Globally distorted -> main features of shape are distorted, local contours preserved

Fields:

1) ground_truth: the base object, in set {camel, hammer, microphone, tshirt, violin, warplane}
2) global_distortion:  the string "scr", indicates an image is scrambled

'''

for filepath in Path(stimuli_directory).glob('*.png'):

    # entire name of image file:
    image_id = filepath.stem

    # Dataset image data, 1-2 from above
    ground_truth = image_id[3:]
    global_distortion = image_id[0:3]

    image_paths[image_id] = filepath
    stimuli.append({
        'image_id': image_id,
        'ground_truth': ground_truth,
        'global_distortion': global_distortion,
    })

stimuli = StimulusSet(stimuli)
stimuli.image_paths = image_paths
stimuli.name = 'kellmen.Baker2018_global_distortion'  # give the StimulusSet an identifier name

# Ensure 6 images in dataset
assert len(stimuli) == 6

# upload to S3
# package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
#                      bucket_name="brainio-brainscore")
