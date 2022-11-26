import itertools
from pathlib import Path
from typing import List

from brainio.stimuli import StimulusSet


def get_sample_stimuli_paths() -> List[Path]:
    stimuli_directory = Path(__file__).parent
    extensions = ['png', 'jpg']
    stimuli_paths = itertools.chain.from_iterable(stimuli_directory.glob(f'*.{extension}') for extension in extensions)
    return sorted(stimuli_paths)


def get_sample_stimulus_set() -> StimulusSet:
    paths = get_sample_stimuli_paths()
    stimulus_ids = [path.stem for path in paths]
    stimulus_set = StimulusSet({
        'stimulus_id': stimulus_ids
    })
    stimulus_set.stimulus_paths = {stimulus_id: path for stimulus_id, path in zip(stimulus_ids, paths)}
    return stimulus_set
