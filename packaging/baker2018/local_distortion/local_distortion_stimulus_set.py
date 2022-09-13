from pathlib import Path
# from brainio.stimuli import StimulusSet
# from brainio.packaging import package_stimulus_set

stimuli = []
image_paths = {}
stimuli_directory = '../local-distortion-images'


'''
Dataset Information:

- From Baker 2018: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006613
- 6 images
- Locally distorted -> edge contours are jagged, sawtooth patterns

Fields:

1) ground_truth: the base object, in set {camel, hammer, microphone, tshirt, violin, warplane}
2) local_distortion:  the string "point", indicates an image is locally distorted, with jagged edges according to experiment,

'''

for filepath in Path(stimuli_directory).glob('*.png'):

    # entire name of image file:
    image_id = filepath.stem
    split_name = filepath.stem.split('_')

    # ensure proper metadata length per image in set
    assert len(split_name) == 2

    # Dataset image data, 1-2 from above
    ground_truth = split_name[0]
    local_distortion = split_name[1]

    image_paths[image_id] = filepath
    stimuli.append({
        'image_id': image_id,
        'ground_truth': ground_truth,
        'local_distortion': local_distortion,
    })

stimuli = StimulusSet(stimuli)
stimuli.image_paths = image_paths
stimuli.name = 'kellmen.Baker2018_local_distortion'  # give the StimulusSet an identifier name

# Ensure 6 images in dataset
assert len(stimuli) == 6

# upload to S3
# package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
#                      bucket_name="brainio-brainscore")
