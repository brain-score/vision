import functools
import dill
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from robustness.model_utils import DummyModel
from robustness.attacker import AttackerModel
from robustness.datasets import DATASETS
from brainscore_vision.model_helpers.s3 import load_weight_file
from brainscore_vision.model_helpers.check_submission import check_models
import torch as ch
import os

def make_and_restore_model(*_, arch, dataset, resume_path=None,
         parallel=False, pytorch_pretrained=False, add_custom_forward=False):
    """
    Makes a model and (optionally) restores it from a checkpoint.

    Args:
        arch (str|nn.Module): Model architecture identifier or otherwise a
            torch.nn.Module instance with the classifier
        dataset (Dataset class [see datasets.py])
        resume_path (str): optional path to checkpoint saved with the 
            robustness library (ignored if ``arch`` is not a string)
        not a string
        parallel (bool): if True, wrap the model in a DataParallel 
            (defaults to False)
        pytorch_pretrained (bool): if True, try to load a standard-trained 
            checkpoint from the torchvision library (throw error if failed)
        add_custom_forward (bool): ignored unless arch is an instance of
            nn.Module (and not a string). Normally, architectures should have a
            forward() function which accepts arguments ``with_latent``,
            ``fake_relu``, and ``no_relu`` to allow for adversarial manipulation
            (see `here`<https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-with-custom-architectures>
            for more info). If this argument is True, then these options will
            not be passed to forward(). (Useful if you just want to train a
            model and don't care about these arguments, and are passing in an
            arch that you don't want to edit forward() for, e.g.  a pretrained model)
    Returns: 
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    """
    if (not isinstance(arch, str)) and add_custom_forward:
        arch = DummyModel(arch)

    classifier_model = dataset.get_model(arch, pytorch_pretrained) if \
                            isinstance(arch, str) else arch

    model = AttackerModel(classifier_model, dataset)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = ch.load(resume_path, pickle_module=dill,map_location="cpu")
        
        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        model.load_state_dict(sd)
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    elif resume_path:
        error_msg = "=> no checkpoint found at '{}'".format(resume_path)
        raise ValueError(error_msg)

    if parallel:
        model = ch.nn.DataParallel(model)
        
    return model, checkpoint

def get_model(name):
    assert name == "imagenet_l2_3_0"
    data_path = "" #os.path.expandvars(args.data)
    dataset = DATASETS['imagenet'](data_path)
    weights_path = load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",
                                   relative_path="imagenet_l2_3_0/imagenet_l2_3_0.pt",
                                   version_id="null",
                                   sha1="cc6e4441abc8ad6d2f4da5db84836e544bfb53fd")
    model, _ = make_and_restore_model(arch='resnet50', dataset=dataset, 
                                      resume_path=weights_path)
    model = model.model.eval()
    # print(model)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == "imagenet_l2_3_0"
    return ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']


def get_bibtex(name):
    return """ """


if __name__ == '__main__':
    check_models.check_base_models(__name__)
