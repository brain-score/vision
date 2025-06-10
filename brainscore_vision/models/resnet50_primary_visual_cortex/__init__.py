from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_primary_visual_cortex'] = lambda: ModelCommitment(
        identifier='resnet50_primary_visual_cortex', 
        activations_model=get_model('resnet50_primary_visual_cortex'), 
        layers=get_layers('resnet50_primary_visual_cortex')
)
