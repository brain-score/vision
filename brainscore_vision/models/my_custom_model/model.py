from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
import torchvision.models as models
import gdown

# This is an example implementation for submitting custom model named my_custom_model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.




def get_model_list():
    return ['fast_2px_step2_eps2_repeat1_trial1_model_best']


def get_model(name):

    trained_model = models.__dict__['resnet50']()
    trained_model = torch.nn.DataParallel(trained_model)

    url = "https://drive.google.com/uc?id=1kNgOmtSrCQnyINVGw_l9vishwaNeqGN4"
    output = "fast_2px_step2_eps2_repeat1_trial1_model_best.pth.tar"
    gdown.download(url, output)

    checkpoint = torch.load("fast_2px_step2_eps2_repeat1_trial1_model_best.pth.tar", map_location=torch.device('cpu'))
    trained_model.load_state_dict(checkpoint['state_dict'])

    trained_model = trained_model.module

    assert name == 'fast_2px_step2_eps2_repeat1_trial1_model_best'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier='fast_2px_step2_eps2_repeat1_trial1_model_best', model=trained_model, preprocessing=preprocessing)
    model = ModelCommitment(identifier='fast_2px_step2_eps2_repeat1_trial1_model_best', activations_model=activations_model,
                            # specify layers to consider
                            layers=['layer1[0].conv3', 'layer1[1].conv3', 'layer1[2].conv3', 
				    'layer2[0].conv3', 'layer2[1].conv3', 'layer2[2].conv3', 'layer2[3].conv3',
				    'layer3[0].conv3', 'layer3[1].conv3', 'layer3[2].conv3', 'layer3[3].conv3', 'layer3[4].conv3', 'layer3[5].conv3',
				    'layer4[0].conv3', 'layer4[1].conv3', 'layer4[2].conv3' ])
    wrapper = PytorchWrapper(identifier='fast_2px_step2_eps2_repeat1_trial1_model_best', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'fast_2px_step2_eps2_repeat1_trial1_model_best'
    return ['layer1[0].conv3', 'layer1[1].conv3', 'layer1[2].conv3', 
	    'layer2[0].conv3', 'layer2[1].conv3', 'layer2[2].conv3', 'layer2[3].conv3',
	    'layer3[0].conv3', 'layer3[1].conv3', 'layer3[2].conv3', 'layer3[3].conv3', 'layer3[4].conv3', 'layer3[5].conv3',
	    'layer4[0].conv3', 'layer4[1].conv3', 'layer4[2].conv3' ]


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
