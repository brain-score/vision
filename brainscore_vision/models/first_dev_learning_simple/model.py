from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

try:
    from .load_standalone_model import load_model
except ImportError:
    from load_standalone_model import load_model
import os

THIS_NAME = "first_dev_learning_simple"

INPUT_SIZE = 100 # TODO change to good value for CLAPP

def get_model(name):
    assert name == THIS_NAME
    here = os.path.dirname(__file__)
    model = load_model(model_path=here, option=0)
    # option=0 is cur_best at epoch 1519 (300 epochs per layer)
    preprocessing = functools.partial(load_preprocess_images, image_size=INPUT_SIZE)
    # the preprocessing: normalizing (with ImageNet values), resizing to given size
    # beware, images will be RGB (not grayscale)
    wrapper = PytorchWrapper(identifier=THIS_NAME, model=model, preprocessing=preprocessing,
                             forward_kwargs={"is_normalized":True, "keep_patches":False, "all_layers":False},
                             batch_size=1)  # Reduced to minimum (64→8→2→1) to prevent OOM
    wrapper.image_size = INPUT_SIZE   # I think this attribute is never used, but anyway
    return wrapper


def get_layers(name):
    # we analyze all 6 layers (at the ouput of VGG_like_Encoder)
    assert name == THIS_NAME
    base = "encoder.{}.dummy"
    return [base.format(i) for i in range(6)]


def get_bibtex(model_identifier):
    return """"""


def get_model_list():
    return [THIS_NAME]


if __name__ == '__main__':
    check_models.check_base_models(__name__)
