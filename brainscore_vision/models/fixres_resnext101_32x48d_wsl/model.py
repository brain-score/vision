from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from fixres.hubconf import load_state_dict_from_url
from fixres.transforms_v2 import get_transforms
from brainscore_vision.model_helpers.activations.pytorch import load_images
import numpy as np
from importlib import import_module
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


def get_model():
    module = import_module('fixres.imnet_evaluate.resnext_wsl')
    model_ctr = getattr(module, 'resnext101_32x48d_wsl')
    model = model_ctr(pretrained=False)  # the pretrained flag here corresponds to standard resnext weights
    pretrained_dict = load_state_dict_from_url('https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNeXt_101_32x48d.pth',
                                               map_location=lambda storage, loc: storage)['model']
    model_dict = model.state_dict()
    for k in model_dict.keys():
        assert ('module.' + k) in pretrained_dict.keys()
        model_dict[k] = pretrained_dict.get(('module.' + k))
    model.load_state_dict(model_dict)

    # preprocessing
    # 320 for ResNeXt:
    # https://github.com/mschrimpf/FixRes/tree/4ddcf11b29c118dfb8a48686f75f572450f67e5d#example-evaluation-procedure
    input_size = 320
    # https://github.com/mschrimpf/FixRes/blob/0dc15ab509b9cb9d7002ca47826dab4d66033668/fixres/imnet_evaluate/train.py#L159-L160
    transformation = get_transforms(input_size=input_size, test_size=input_size,
                                    kind='full', need=('val',),
                                    # this is different from standard ImageNet evaluation to show the whole image
                                    crop=False,
                                    # no backbone parameter for ResNeXt following
                                    # https://github.com/mschrimpf/FixRes/blob/0dc15ab509b9cb9d7002ca47826dab4d66033668/fixres/imnet_evaluate/train.py#L154-L156
                                    backbone=None)
    transform = transformation['val']

    def load_preprocess_images(image_filepaths):
        images = load_images(image_filepaths)
        images = [transform(image) for image in images]
        images = [image.unsqueeze(0) for image in images]
        images = np.concatenate(images)
        return images

    wrapper = PytorchWrapper(identifier='resnext101_32x48d_wsl', model=model, preprocessing=load_preprocess_images,
                             batch_size=4)  # doesn't fit into 12 GB GPU memory otherwise
    wrapper.image_size = input_size
    return wrapper


def get_layers(name):
    return (['conv1'] +
            # note that while relu is used multiple times, by default the last one will overwrite all previous ones
            [f"layer{block + 1}.{unit}.relu"
             for block, block_units in enumerate([3, 4, 23, 3]) for unit in range(block_units)] +
            ['avgpool'])
