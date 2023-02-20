import logging

import brainscore
from candidate_models.model_commitments import brain_translated_pool

_logger = logging.getLogger(__name__)

def get_activations(model, layers, stimulus_set):
    stimuli = brainscore.get_stimulus_set(stimulus_set)
    return model(stimuli, layers=layers)
