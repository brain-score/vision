
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, MODEL_CONFIGS, get_layers

network = 'alexnet'
for keyword in ['imagenet_trained', 'less_variation', 'no_variation', 'z_axis', 'wo_shading','wo_shadows']:
    for iteration in range(1,6):
        model_registry[f'{keyword}_{network}_{iteration}=1'] = lambda: ModelCommitment(identifier=f'{keyword}_{network}_{iteration}=1', activations_model=get_model(f'{keyword}_{network}_{iteration}=1'), layers=get_layers(f'{keyword}_{network}_{iteration}=1'))