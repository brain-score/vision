import copy

import numpy as np
import pandas as pd

from brainio_base.assemblies import walk_coords, array_is_element


def repeat_trials(stimulus_set, number_of_trials):
    # maintain metadata
    assert not hasattr(stimulus_set, 'repetition')
    meta = {meta_key: copy.deepcopy(getattr(stimulus_set, meta_key)) for meta_key in stimulus_set._metadata
            if not callable(getattr(stimulus_set, meta_key))}  # don't copy functions
    # repeat stimulus set
    num_stimuli = len(stimulus_set)
    assert isinstance(number_of_trials, int)
    stimulus_set = pd.concat([stimulus_set] * number_of_trials)
    stimulus_set['repetition'] = list(range(number_of_trials)) * num_stimuli
    # re-attach meta
    for meta_key, meta_value in meta.items():
        setattr(stimulus_set, meta_key, meta_value)
    # we use a different identifier from the original identifier here so stochastic model will produce different
    # activations for different trials. Non-stochastic model might have to be run multiple times but in practice the
    # same stimulus set usually has the same number of trials.
    if hasattr(stimulus_set, 'identifier') and stimulus_set.identifier is not None:
        stimulus_set.identifier = stimulus_set.identifier + f'-{number_of_trials}trials'
    return stimulus_set


def average_trials(assembly):
    non_repetition_coords = [coord for coord, dim, values in walk_coords(assembly['presentation'])
                             if array_is_element(dim, 'presentation') and coord != 'repetition']
    grouped = assembly.multi_groupby(non_repetition_coords)
    if np.issubdtype(assembly.dtype, np.number):
        return grouped.mean('presentation')
    else:  # for non-numbers take majority
        return grouped.max('presentation')
