from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers, get_bibtex

model_registry['saycam_resnext'] = lambda: ModelCommitment(
    identifier='saycam_resnext', 
    activations_model=get_model('saycam_resnext'), 
    layers=get_layers('saycam_resnext')
)
