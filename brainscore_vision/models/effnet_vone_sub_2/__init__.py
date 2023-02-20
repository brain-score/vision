from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb1_VOne_untrained_324x288'] = ModelCommitment(identifier='effnetb1_VOne_untrained_324x288', activations_model=get_model('effnetb1_VOne_untrained_324x288'), layers=get_layers('effnetb1_VOne_untrained_324x288'))
