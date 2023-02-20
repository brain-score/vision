import functools
import os

import torch
import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
import scp_client


# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models

from networks import network_utils, resnet

# https://stackoverflow.com/a/39225039/2225200
import requests

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    # mobilenet_v3_small, mobilenet_v3_small-LC, mobilenet_v3_small-LC-conv-init,
    # mobilenet_v3_small-LC-conv-1x1, mobilenet_v3_small-LC-conv-1x1-init
    # mobilenet_v3_small-LC-dyn-1x1
    return ['mobilenet_v3_small-LC']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    model = torchvision.models.mobilenet_v3_small()

    saved_name = 'lc'  # conv, lc, lc_conv_init, lc_1x1, lc_1x1_conv_init, lc_dyn

    class Args:
        def __init__(self):
            self.dataset = 'ImageNet'
            self.input_scale = 1
            self.n_first_conv = 0
            self.conv_1x1 = '1x1' in saved_name
            self.locally_connected_deviation_eps = -1
            self.old_1x1 = saved_name == 'lc'
            self.dynamic_1x1 = 'dyn' in saved_name
            self.dynamic_NxN = False
            self.dynamic_sharing_hebb_lr = 0.0
            self.dynamic_sharing_b_freq = 1
    args = Args()

    if saved_name != 'conv':
        network_utils.convert_network_to_locally_connected(model, args)

    model_path = './%s.pt' % saved_name  #'./%s.pt' % name  # os.path.join(os.path.dirname(__file__), '/%s.pt' % name)
    # file_id = '1UfqWCRn1bln-BOWO06qdTRkcd0hqdXg9'
    # #'1MncmTiaw_nJwMnQt9hPLg3XB25Vvw18p'  # lc
    # #'1RUGiO5lMQUVMc5rKyoQFObFYEB8U75R5'  # conv
    if not os.path.isfile(model_path):
        key_path = ''
        if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), '/../key')):
            key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/../key')
        elif os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), '/key')):
            key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/key')
        elif os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), './key')):
            key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './key')
        elif os.path.isfile('./key'):
            key_path = './key'
        elif os.path.isfile('./../key'):
            key_path = './../key'
        elif os.path.isfile('./brainscore-lc/lc/key'):
            key_path = './brainscore-lc/lc/key'
        elif os.path.isfile('./lc/key'):
            key_path = './lc/key'
        else:
            raise FileNotFoundError('Cannot find key in %s' % os.getcwd())

        client = scp_client.RemoteClient('ssh.swc.ucl.ac.uk', 'romanp', 'brainscore',
                                         key_path,
                                         './Studies/UCL/research_code/plausible-conv-saved-models/')
        client.download_file('./Studies/UCL/research_code/plausible-conv-saved-models/%s.pt' % saved_name)
        # download_file_from_google_drive(file_id, model_path)

    # os.chmod(model_path, 0o777)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


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
    layers = ['features.0.2']


    for layer in range(1, 12):
        layers.append('features.%d.block.0.2' % layer)
        if layer != 1:
            layers.append('features.%d.block.1.2' % layer)
        if layer < 4:
            layers.append('features.%d.block.2.2' % layer)
        else:
            layers.append('features.%d.block.3.2' % layer)
    layers.append('features.12.2')
    layers.append('classifier.1')
    layers.append('classifier.3')
    return layers


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@article{pogodin2021towards,
                title={Towards Biologically Plausible Convolutional Networks},
                author={Pogodin, Roman and Mehta, Yash and Lillicrap, Timothy P and Latham, Peter E},
                journal={arXiv preprint arXiv:2106.13031},
                year={2021},
                url={https://arxiv.org/pdf/2106.13031.pdf}
                }"""


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
