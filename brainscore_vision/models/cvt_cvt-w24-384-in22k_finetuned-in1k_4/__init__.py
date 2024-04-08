from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['cvt_cvt-w24-384-in22k_finetuned-in1k_4'] = lambda: ModelCommitment(identifier='cvt_cvt-w24-384-in22k_finetuned-in1k_4', activations_model=get_model('cvt_cvt-w24-384-in22k_finetuned-in1k_4'), layers=get_layers('cvt_cvt-w24-384-in22k_finetuned-in1k_4'))
