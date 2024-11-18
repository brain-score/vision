from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['cvt_cvt-21-384-in22k_finetuned-in1k_4_LucyV4'] = \
    lambda: ModelCommitment(identifier='cvt_cvt-21-384-in22k_finetuned-in1k_4_LucyV4',
                            activations_model=get_model('cvt_cvt-21-384-in22k_finetuned-in1k_4_LucyV4'),
                            layers=get_layers('cvt_cvt-21-384-in22k_finetuned-in1k_4_LucyV4'))