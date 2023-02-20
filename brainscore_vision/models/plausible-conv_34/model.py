import functools
import os

import torch
import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
import scp_client
from torchvision.models.resnet import load_state_dict_from_url as load_state_dict_from_url

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models

from networks import network_utils, resnet, cornet_s

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
    # + resnet-18-LC
    # + resnet-18-LC_w_sh_1_iter
    # + resnet-18-LC_w_sh_10_iter
    # + resnet-18-LC_w_sh_100_iter
    # + resnet-18-LC_conv_init
    # + resnet-18-LC_w_sh_1_iter_conv_init
    # + resnet-18-LC_w_sh_10_iter_conv_init
    # + resnet-18-LC_w_sh_100_iter_conv_init
    # + resnet-18-LC_d_w_sh_1x1_conv_init
    # + resnet-18_test (just r18 pre-trained from pytorch)
    # + 0.25xCORNet-S-LC
    # + 0.25xCORNet-S-LC_conv_init
    # + 0.25xCORNet-S-LC_d_w_sh_1x1_conv_init
    # (test) just cornet
    # + resnet-18-LC_1st_conv
    # + resnet-18-LC_1st_conv_conv_init
    # + resnet-10-two-blocks
    # + resnet-10-two-blocks-LC
    # + resnet-10m-two-blocks
    # + resnet-10m-two-blocks-LC
    # ? resnet-18-LC_d_w_sh_conv_init

    model_list = ['resnet-18-LC_untrained', 'resnet-18_untrained']
    # ['resnet-18-LC', 'resnet-18-LC_w_sh_1_iter', 'resnet-18-LC_w_sh_10_iter',
                  # 'resnet-18-LC_w_sh_100_iter',
                  # 'resnet-18-LC_1st_conv',
                  # # 'resnet-18-LC_d_w_sh_1x1_conv_init',
                  # ]
    #['resnet-10W-two-blocks-LC', 'resnet-10Wm-two-blocks-LC']
    #  ['resnet-10W-two-blocks-LC_prelim3', 'resnet-10Wm-two-blocks-LC_prelim3']
    #['resnet-18-LC_d_w_sh_conv_init_prelim', 'resnet-18-LC_d_w_sh_conv_init_prelim2']

    # for i in range(l2en(model_list)):
    #     model_list[i] += '_conv_init'
    # #
    # for i in range(len(model_list)):
    #     model_list[i] += '_m'

    return model_list


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    if name[-2:] == '_m':
        name = name[:-2]
    if 'resnet-10-two-block' in name:
        model = resnet.resnet10_two_block(width_per_group=64, brain_score=True)
        folder = './Studies/UCL/research_code/plausible-conv-saved-models-r10'
    elif 'resnet-10m-two-block' in name:
        model = resnet.resnet10_two_block_m(width_per_group=64, brain_score=True)
        folder = './Studies/UCL/research_code/plausible-conv-saved-models-r10'
    elif 'resnet-10W-two-block' in name:
        model = resnet.resnet10_two_block(width_per_group=96, conv1_width=8, brain_score=True)
        folder = './Studies/UCL/research_code/plausible-conv-saved-models-r10'
    elif 'resnet-10Wm-two-block' in name:
        model = resnet.resnet10_two_block_m(width_per_group=96, conv1_width=8, brain_score=True)
        folder = './Studies/UCL/research_code/plausible-conv-saved-models-r10'
    elif '0.5x_resnet' in name:
        model = resnet.resnet18(width_per_group=32, brain_score=True)
        folder = './Studies/UCL/research_code/plausible-conv-saved-models'
    elif 'resnet' in name:
        model = resnet.resnet18(width_per_group=64, brain_score=True)
        folder = './Studies/UCL/research_code/plausible-conv-saved-models-r18'
    elif 'COR' in name:
        model = cornet_s.CORnet_S(scale=1)
        folder = './Studies/UCL/research_code/plausible-conv-saved-models-cornet'
    elif 'mobilenet' in name:
        model = torchvision.models.mobilenet_v3_small()
        folder = './Studies/UCL/research_code/plausible-conv-saved-models-mbv3'
    else:
        raise NotImplementedError(name)

    class Args:
        def __init__(self):
            self.dataset = 'ImageNet'
            self.input_scale = 1
            self.n_first_conv = 0
            if '1st_conv' in name:
                self.n_first_conv = 1
            self.conv_1x1 = False
            self.locally_connected_deviation_eps = -1
            self.dynamic_1x1 = 'd_w_sh' in name
            self.dynamic_NxN = 'd_w_sh' in name
            self.dynamic_sharing_hebb_lr = 0.0
            self.dynamic_sharing_b_freq = 1
    args = Args()

    if 'LC' in name:
        network_utils.convert_network_to_locally_connected(model, args)

    if not 'untrained' in name:
        if name == 'resnet-18_test':
            model.load_state_dict(load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth'))
        else:
            saved_name = name # 'resnet-18_LC' #_w_sh_1_iter' # 'cornet_s_512_1x1_conv_start'
            model_path = './%s.pt' % saved_name  #'./%s.pt' % name  # os.path.join(os.path.dirname(__file__), '/%s.pt' % name)
            # file_id = '1UfqWCRn1bln-BOWO06qdTRkcd0hqdXg9'
            # #'1MncmTiaw_nJwMnQt9hPLg3XB25Vvw18p'  # lc_wsh_100
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
                elif os.path.isfile('./brainscore-submission/plausible-conv/key'):
                    key_path = './brainscore-submission/plausible-conv/key'
                elif os.path.isfile('./plausible-conv/key'):
                    key_path = './plausible-conv/key'
                else:
                    raise FileNotFoundError('Cannot find key in %s' % os.getcwd())

                client = scp_client.RemoteClient('ssh.swc.ucl.ac.uk', 'romanp', 'brainscore',
                                                 key_path, folder)
                client.download_file('%s/%s.pt' % (folder, saved_name))
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
    if 'COR' in name:
        layers = ['V1.output', 'V2.output', 'V4.output', 'IT.output', 'decoder.output']
    else:
        layers = ['relu']

        n_blocks = 2 if 'two-block' in name else 4
        for layer in range(1, n_blocks + 1):
            for block in range(2):
                layers.append('layer%d.%d.relu1' % (layer, block))
                layers.append('layer%d.%d.relu2' % (layer, block))
        layers.append('logits')
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
    # return """@inproceedings{KubiliusSchrimpf2019CORnet,
    #             abstract = {Deep convolutional artificial neural networks (ANNs) are the leading class of candidate models of the mechanisms of visual processing in the primate ventral stream. While initially inspired by brain anatomy, over the past years, these ANNs have evolved from a simple eight-layer architecture in AlexNet to extremely deep and branching architectures, demonstrating increasingly better object categorization performance, yet bringing into question how brain-like they still are. In particular, typical deep models from the machine learning community are often hard to map onto the brain's anatomy due to their vast number of layers and missing biologically-important connections, such as recurrence. Here we demonstrate that better anatomical alignment to the brain and high performance on machine learning as well as neuroscience measures do not have to be in contradiction. We developed CORnet-S, a shallow ANN with four anatomically mapped areas and recurrent connectivity, guided by Brain-Score, a new large-scale composite of neural and behavioral benchmarks for quantifying the functional fidelity of models of the primate ventral visual stream. Despite being significantly shallower than most models, CORnet-S is the top model on Brain-Score and outperforms similarly compact models on ImageNet. Moreover, our extensive analyses of CORnet-S circuitry variants reveal that recurrence is the main predictive factor of both Brain-Score and ImageNet top-1 performance. Finally, we report that the temporal evolution of the CORnet-S "IT" neural population resembles the actual monkey IT population dynamics. Taken together, these results establish CORnet-S, a compact, recurrent ANN, as the current best model of the primate ventral visual stream.},
    #             archivePrefix = {arXiv},
    #             arxivId = {1909.06161},
    #             author = {Kubilius, Jonas and Schrimpf, Martin and Hong, Ha and Majaj, Najib J. and Rajalingham, Rishi and Issa, Elias B. and Kar, Kohitij and Bashivan, Pouya and Prescott-Roy, Jonathan and Schmidt, Kailyn and Nayebi, Aran and Bear, Daniel and Yamins, Daniel L. K. and DiCarlo, James J.},
    #             booktitle = {Neural Information Processing Systems (NeurIPS)},
    #             editor = {Wallach, H. and Larochelle, H. and Beygelzimer, A. and D'Alch{\'{e}}-Buc, F. and Fox, E. and Garnett, R.},
    #             pages = {12785----12796},
    #             publisher = {Curran Associates, Inc.},
    #             title = {{Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs}},
    #             url = {http://papers.nips.cc/paper/9441-brain-like-object-recognition-with-high-performing-shallow-recurrent-anns},
    #             year = {2019}
    #             }"""


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
