import functools
from brainscore_vision.model_helpers.activations.pytorch import load_images, preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

def load_preprocess_images(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    images = images[:, :, None, ...]
    return images


def preprocessing(stimuli_paths):
    # following https://github.com/facebookresearch/omnivore/blob/main/inference_tutorial.ipynb
    stimuli = [Image.open(image_path).convert("RGB") for image_path in stimuli_paths]
    image_transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    stimuli = [image_transform(image) for image in stimuli]
    # The model expects inputs of shape: B x C x T x H x W
    stimuli = [image[:, None, ...] for image in stimuli]
    return stimuli

model = torch.hub.load("facebookresearch/omnivore:main", model="omnivore_swinB", force_reload=True)

def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'omnivore_swinB'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='name', model=model, preprocessing=preprocessing,forward_kwargs={"input_type":"image"})
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'omnivore_swinB'
    model_blocks = {
        "omnivore_swinT": [2, 2, 6, 2],
        "omnivore_swinS": [2, 2, 18, 2],
        "omnivore_swinB": [2, 2, 18, 2],
        "omnivore_swinB_imagenet21k": [2, 2, 18, 2],
        "omnivore_swinL_imagenet21k": [2, 2, 18, 2],
    }
    blocks = model_blocks[name]
    return ['trunk.pos_drop'] + [f'trunk.layers.{layer}.blocks.{block}'
                                 for layer, num_blocks in enumerate(blocks) for block in range(num_blocks)]


def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@inproceedings{girdhar2022omnivore,
                 title={{Omnivore: A Single Model for Many Visual Modalities}},
                 author={Girdhar, Rohit and Singh, Mannat and Ravi, Nikhila and van der Maaten, Laurens and Joulin, Armand and Misra, Ishan},
                 booktitle={CVPR},
                 year={2022},
                 url={https://doi.org/10.48550/arXiv.2201.08377}
               }"""


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)