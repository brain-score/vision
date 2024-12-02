
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, MODEL_CONFIGS, get_layers
network = 'alexnet'
for keyword in ['imagenet_trained', 'less_variation', 'no_variation', 'z_axis', 'wo_shading','wo_shadows', 'ambient', 'textures', 'no_specular']:
    for iteration in range(1,6):
        model_registry[f'{network}_{keyword}_iteration={iteration}'] = lambda: ModelCommitment(identifier=f'{network}_{keyword}_iteration={iteration}', activations_model=get_model('{network}_{keyword}_iteration={iteration}'), layers=get_layers(f'{network}_{keyword}_iteration={iteration}'))


