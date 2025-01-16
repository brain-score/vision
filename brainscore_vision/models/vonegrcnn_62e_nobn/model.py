from brainscore_vision.model_helpers.check_submission import check_models
import torch
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.s3 import load_weight_file
from .helpers.vongrcnn_helpers import VOneNet, grcnn55BackEnd
import torch.nn as nn
from collections import OrderedDict

device = "cpu"
model_identifier = 'vonegrcnn_62e_nobn'


###DEFINE YOUR CUSTOM MODEL HERE

# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'vonegrcnn_62e_nobn'
    # link the custom model to the wrapper object(activations_model above):
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    vone_block, bottleneck = VOneNet()
    model_back_end = grcnn55BackEnd()

    model = nn.Sequential(OrderedDict([
                    ('vone_block', vone_block),
                    ('bottleneck', bottleneck),
                    ('model', model_back_end),
                ]))
        
    model = nn.Sequential(OrderedDict([('module',model)]))
    weights_path = load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",
                                    relative_path="vonegrcnn_62e/model_best.pth",
                                    version_id="null",
                                    sha1="66f5319888ebd146565fb45144afa92d8a2bef3b")
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(device)

    # get an activations model from the Pytorch Wrapper
    activations_model = PytorchWrapper(identifier=model_identifier, model= model,
                                    preprocessing=preprocessing)
    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


# get_layers method to tell the code what layers to consider. If you are submitting a custom
# model, then you will most likely need to change this method's return values.
def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """

    # quick check to make sure the model is the correct one:
    assert name == 'vonegrcnn_62e_nobn'
    all_layers = ['module',
    'module.vone_block',
    'module.vone_block.simple_conv_q0',
    'module.vone_block.simple_conv_q1',
    'module.vone_block.simple',
    'module.vone_block.complex',
    'module.vone_block.gabors',
    'module.vone_block.noise',
    'module.vone_block.output',
    'module.bottleneck',
    'module.model',
    'module.model.layer1',
    'module.model.layer2',
    'module.model.layer3',
    'module.model.layer4',
    'module.model.layer1.conv_f',
    'module.model.layer2.conv_f',
    'module.model.layer3.conv_f',
    'module.model.layer4.conv_f',
    'module.model.layer1.d_conv_1e',
    'module.model.layer2.d_conv_1e',
    'module.model.layer3.d_conv_1e',
    'module.model.layer1.iter_g_3.1',
    'module.model.layer2.iter_g_3.1',
    'module.model.layer3.iter_g_4.1',
    'module.model.layer4.iter_g_3.1',
    'module.model.lastact',
    'module.model.lastact.0',
    'module.model.lastact.1',
    'module.model.avgpool',
    'module.model.classifier']
    # returns the layers you want to consider
    return  all_layers


# Bibtex Method. For submitting a custom model, you can either put your own Bibtex if your
# model has been published, or leave the empty return value if there is no publication to refer to.
def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    # from pytorch.py:
    return ''


# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
