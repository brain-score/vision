from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['simple_spiking_model'] = \
    lambda: ModelCommitment(identifier='simple_spiking_model', 
                            activations_model=get_model('simple_spiking_model'), 
                            layers=get_layers('simple_spiking_model'))