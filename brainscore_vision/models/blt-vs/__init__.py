from pretrained import (
    register_model,
    register_aliases,
    clear_models_and_aliases,
 
)
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from model import get_model, LAYERS
from model import BLT_VS


clear_models_and_aliases(BLT_VS)

register_model(
        BLT_VS,
        'blt_vs',
        'https://zenodo.org/records/14223659/files/blt_vs.zip',
        '36d74a367a261e788028c6c9caa7a5675fee48e938a6b86a6c62655b23afaf53'
    )

register_aliases(BLT_VS, 'blt_vs', 'blt_vs')
    
model_registry['blt_vs'] = lambda: ModelCommitment(
    identifier='blt_vs',
    activations_model=get_model(),
    layers=LAYERS)

