from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb1_VOneGrad_e3_392x348'] = ModelCommitment(identifier='effnetb1_VOneGrad_e3_392x348', activations_model=get_model('effnetb1_VOneGrad_e3_392x348'), layers=get_layers('effnetb1_VOneGrad_e3_392x348'))
