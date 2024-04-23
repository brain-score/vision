import timm 
from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment


# This is an example implementation for submitting custom model named my_custom_model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.



def get_model_list():
    return ['EVA-02']


def get_model(name):
    assert name == 'EVA-02'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    
    # model = ModelCommitment(identifier='EVA-02', activations_model=activations_model,
    #                         # specify layers to consider
    # #                         layers=['conv1', 'relu1', 'relu2'])

    model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)

    activations_model = PytorchWrapper(identifier='EVA-02', model=model, preprocessing=preprocessing)
    
    wrapper = PytorchWrapper(identifier='EVA-02', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'EVA-02'
    return ['model.blocks[0].mlp', 'model.blocks[3].mlp', 'model.blocks[6].mlp',
            'model.blocks[9].mlp', 'model.blocks[12].mlp', 'model.blocks[15].mlp',
             'model.blocks[18].mlp', 'model.blocks[21].mlp', 'model.blocks[23].mlp']


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
