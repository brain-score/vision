import copy
from collections import defaultdict

import pandas as pd


def infer_trials_from_parameter_or_assembly(number_of_trials, assembly):
    if number_of_trials is None:
        assert hasattr(assembly, 'repetitions'), "number_of_trials parameter not specified " \
                                                "and assembly['repetitions'] missing"
        return {row.image_id: len(row.repetition) for row in assembly.groupby('image_id')}
    else:
        assert not hasattr(assembly, 'repetitions'), "number_of_trials can be specified through " \
                                                    "either the parameter or assembly['repetitions'], but not both"
        return number_of_trials


def repeat_trials(stimulus_set, number_of_trials):
    assert not hasattr(stimulus_set, 'repetition')
    meta = {meta_key: copy.deepcopy(getattr(stimulus_set, meta_key)) for meta_key in stimulus_set._metadata
            if not callable(getattr(stimulus_set, meta_key))}  # don't copy functions
    # number_of_trials can either be an integer or some per-image assignment
    if isinstance(number_of_trials, int):
        identifier_suffix = f"{number_of_trials}"
        stimulus_set = pd.concat([stimulus_set] * number_of_trials)
        repetitions, repetition_counter = [], defaultdict(lambda: 0)
        for _, row in stimulus_set.iterrows():
            repetitions.append(repetition_counter[row['image_id']])
            repetition_counter[row['image_id']] += 1
        stimulus_set['repetition'] = repetitions
    else:
        identifier_suffix = hash(number_of_trials)
        raise NotImplementedError()

    for meta_key, meta_value in meta.items():
        setattr(stimulus_set, meta_key, meta_value)
    # we use different identifiers here so stochastic model will produce different activations for different trials.
    # Non-stochastic model might have to be run multiple times but in practice the same stimulus set usually has the
    # same number of trials
    stimulus_set.identifier = stimulus_set.identifier + f'-{identifier_suffix}trials'
    return stimulus_set


def average_trials(assembly):
    return assembly.mean('repetition')
