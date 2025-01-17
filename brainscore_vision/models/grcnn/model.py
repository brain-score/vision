from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.s3 import load_weight_file
import torch
from .helpers.helpers import grcnn55, device


def get_model(name):
    assert name == 'grcnn'
    weights_path = load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",
                                    relative_path="grcnn/checkpoint_params_grcnn55.pt",
                                    version_id="null",
                                    sha1="20fb844e72f21aeb257c053adb2b645bc954839e")
    checkpoint = torch.load(weights_path, map_location=device)
    model_ft = grcnn55()
    model_ft.load_state_dict(checkpoint)
    model_ft = model_ft.to(device)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='grcnn', model=model_ft, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'grcnn'
    layers = ['conv1', 'conv2', 'layer1', 'layer1.conv_g_f', 'layer1.iter_1.8',
              'layer1.iter_g_1.1', 'layer1.iter_2.8', 'layer1.iter_g_2.1', 'layer1.iter_3.3', 'layer1.iter_3.8',
              'layer1.iter_g_3.1', 'layer1.d_conv_1', 'layer1.d_conv_3', 'layer2.conv_g_r',
              'layer2.iter_1.8', 'layer2.iter_g_1.1', 'layer2.iter_2.8', 'layer2.iter_g_2.1', 'layer2.iter_3.8',
              'layer2.iter_g_3.1', 'layer3.conv_f', 'layer3.conv_g_r', 'layer3.iter_1.8', 'layer3.iter_g_1.1',
              'layer3.iter_2.8', 'layer3.iter_g_2.1', 'layer3.iter_3.8', 'layer3.iter_g_3.1', 'layer3.iter_4.8',
              'layer3.iter_g_4.1', 'layer3.d_conv_1e', 'layer4.conv_g_r', 'layer4.iter_1.8', 'layer4.iter_g_1.1',
              'layer4.iter_2.8', 'layer4.iter_g_2.1', 'layer4.iter_3.8', 'layer4.iter_g_3.1',
              'lastact.1', 'classifier']

    return layers


def get_bibtex(model_identifier):
    return '''@misc{cheng2020grcnngraphrecognitionconvolutional,
      title={GRCNN: Graph Recognition Convolutional Neural Network for Synthesizing Programs from Flow Charts}, 
      author={Lin Cheng and Zijiang Yang},
      year={2020},
      eprint={2011.05980},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2011.05980}, 
}'''


if __name__ == '__main__':
    check_models.check_base_models(__name__)
