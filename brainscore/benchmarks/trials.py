import copy
from collections import defaultdict

import pandas as pd

from brainio_base.assemblies import walk_coords


def repeat_trials(stimulus_set, number_of_trials):
    # maintain metadata
    assert not hasattr(stimulus_set, 'repetition')
    meta = {meta_key: copy.deepcopy(getattr(stimulus_set, meta_key)) for meta_key in stimulus_set._metadata
            if not callable(getattr(stimulus_set, meta_key))}  # don't copy functions
    # repeat stimulus set
    assert isinstance(number_of_trials, int)
    stimulus_set = pd.concat([stimulus_set] * number_of_trials)
    repetitions, repetition_counter = [], defaultdict(lambda: 0)
    for _, row in stimulus_set.iterrows():
        repetitions.append(repetition_counter[row['image_id']])
        repetition_counter[row['image_id']] += 1
    stimulus_set['repetition'] = repetitions
    # re-attach meta
    for meta_key, meta_value in meta.items():
        setattr(stimulus_set, meta_key, meta_value)
    # we use a different identifier from the original identifier here so stochastic model will produce different
    # activations for different trials. Non-stochastic model might have to be run multiple times but in practice the
    # same stimulus set usually has the same number of trials.
    stimulus_set.identifier = stimulus_set.identifier + f'-{number_of_trials}trials'
    return stimulus_set


def average_trials(assembly):
    non_repetition_coords = [coord for coord, dim, values in walk_coords(assembly['presentation'])
                             if coord != 'repetition']
    return assembly.multi_groupby(non_repetition_coords).mean('presentation')
