from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

"""
Takes in stimuli (directory path) that were used in a specific MTurk run, creates a stimulus set, tests them, and
uploads that stimulus set to BrainIO.

Sample stimuli from dataset: 
    image_657358285825823573.jpg

Stimuli names are a concatenation of the following fields, seperated via the _ character:
    1) str, stimuli type: most likely will be the string literal 'image', but could be 'video' in the future.
    2) stimuli hash: a SHA-256 pixel hash of the stimuli, unique to the stimuli. Stimuli are linked to metadata
     via assembly that uses this stimulus_set.
"""


def path_to_stimulus_set(stimulus_set_name: str, stimuli_path: str) -> StimulusSet:
    """

    :param stimulus_set_name: desired name of the stimulus set
    :param stimuli_path: file path from root where the stimuli to be packaged are located
    :return: a Stimulus Set object of the stimuli
    """
    stimuli = []
    image_paths = {}

    for filepath in Path(stimuli_path).glob('*.png'):
        # entire name of stimuli file:
        stimulus_id = filepath.stem

        components = stimulus_id.split("_")
        stimulus_type = components[0]
        stimulus_hash = components[1]

        image_paths[stimulus_id] = filepath
        stimuli.append({
            'stimulus_id': stimulus_id,
            'stimulus_type': stimulus_type,
            'stimulus_hash': stimulus_hash,
        })

    stimuli = StimulusSet(stimuli)
    stimuli.stimulus_paths = image_paths
    stimuli.name = stimulus_set_name  # give the StimulusSet an identifier name

    # upload to S3
    # package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
    #                      bucket_name="brainio-brainscore")

    return stimuli
