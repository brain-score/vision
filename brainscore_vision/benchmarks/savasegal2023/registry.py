import os
from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet


STORE_DIR = "/home/ytang/workspace/data/event-segmentation/origin"
MOVIES = ["Defeat.mp4", "Growth.mp4", "Iteration.mp4", "Lemonade.mp4"]

def load_dataset(identifier='SavaSegal2023-fMRI'):

    assembly = NeuroidAssembly()
    stimulus_ids = MOVIES
    stimulus_paths = [os.path.join(STORE_DIR, movie) for movie in MOVIES]

    # make stimulus set
    stimulus_set = StimulusSet([{'stimulus_path': path, "stimulus_id": i} 
                                for i, path in zip(stimulus_ids, stimulus_paths)])
    stimulus_set.identifier = f'SavaSegal-fMRI'
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly
