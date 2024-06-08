from brainscore_vision.model_helpers.check_submission import check_models
import functools
import numpy as np
import torch
from transformers import AutoFeatureExtractor, CvtForImageClassification, CLIPVisionModel, CLIPProcessor, CLIPModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image


"""
Template module for a base model submission to brain-score
"""

class PytorchWrapperFixed(PytorchWrapper):
    @staticmethod
    def _tensor_to_numpy(output):
        # Verificar si la salida es un tuple y tomar solo el primer elemento
        if isinstance(output, tuple):
            output = output[0]
        return output.cpu().data.numpy()

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if isinstance(output, tuple):
                output = output[0]
            target_dict[name] = PytorchWrapperFixed._tensor_to_numpy(output)
        hook = layer.register_forward_hook(hook_function)
        return hook

def get_model_list():
  return ['model-lecs-v1.0.1']

def get_model(name):
    assert name == 'model-lecs-v1.0.1'
    # https://huggingface.co/models?sort=downloads&search=cvt
    image_size = 224
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
    preprocessing = functools.partial(load_preprocess_images, processor=processor, image_size=image_size)
    wrapper = PytorchWrapperFixed(identifier=name, model=model, preprocessing=preprocessing)
    #wrapper = PytorchWrapper(identifier='model-lecs', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224

    return wrapper


def get_layers(name):
    assert name == 'model-lecs-v1.0.1'
    layers = []
    
    # Añadir las capas del modelo CLIPVisionModel
    layers += [f'vision_model.encoder.layers.{i}' for i in [2, 3, 6, 8]]
    # Ejemplo: puedes elegir algunas capas específicas si lo deseas
    # layers = ['vision_model.encoder.layers.3', 'vision_model.post_layernorm']
    
    return layers


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@article{DBLP:journals/corr/abs-2103-15808,
  author       = {Haiping Wu and
                  Bin Xiao and
                  Noel Codella and
                  Mengchen Liu and
                  Xiyang Dai and
                  Lu Yuan and
                  Lei Zhang},
  title        = {CvT: Introducing Convolutions to Vision Transformers},
  journal      = {CoRR},
  volume       = {abs/2103.15808},
  year         = {2021},
  url          = {https://arxiv.org/abs/2103.15808},
  eprinttype    = {arXiv},
  eprint       = {2103.15808},
  timestamp    = {Tue, 18 Oct 2022 08:35:30 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2103-15808.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}"""


def load_preprocess_images(image_filepaths, image_size, processor=None, **kwargs):
    images = load_images(image_filepaths)
    # images = [<PIL.Image.Image image mode=RGB size=400x400 at 0x7F8654B2AC10>, ...]
    images = [image.resize((image_size, image_size)) for image in images]
    if processor is not None:
        images = [processor(images=image, return_tensors="pt", **kwargs) for image in images]
        if len(images[0].keys()) != 1:
            raise NotImplementedError(f'unknown processor for getting model {processor}')
        assert list(images[0].keys())[0] == 'pixel_values'
        images = [image['pixel_values'] for image in images]
        images = torch.cat(images)
        images = images.cpu().numpy()
    else:
        images = preprocess_images(images, image_size=image_size, **kwargs)
    return images


def load_images(image_filepaths):
    return [load_image(image_filepath) for image_filepath in image_filepaths]


def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper() \
                and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image


def preprocess_images(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images


def torchvision_preprocess_input(image_size, **kwargs):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        torchvision_preprocess(**kwargs),
    ])


def torchvision_preprocess(normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])


def create_static_video(image, num_frames, normalize_0to1=False, channel_dim=3):
    '''
    Create a static video with the same image in all frames.
    Args:
        image (PIL.Image.Image): Input image.
        num_frames (int): Number of frames in the video.
    Returns:
        result (np.ndarray): np array of frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    for _ in range(num_frames):
        frame = np.array(image)
        if normalize_0to1:
            frame = frame / 255.
        if channel_dim == 1:
            frame = frame.transpose(2, 0, 1)
        frames.append(frame)
    return np.stack(frames)


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)


'''
Below Notes are from the original model file from Brain-Score 1.0, and 
kept in this file for posterity. 
'''


"""
Notes on the error:
- 'channel_x' key error: 
# 'embeddings.patch_embeddings.projection',
https://github.com/search?q=repo%3Abrain-score%2Fmodel-tools%20channel_x&type=code
"""