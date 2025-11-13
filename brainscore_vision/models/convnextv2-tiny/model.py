from brainscore_vision.model_helpers.check_submission import check_models
import functools
import numpy as np
import torch
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images


"""
Template module for a base model submission to brain-score
"""

def get_model(name):
    assert name == 'convnextv2-tiny'
    # https://huggingface.co/models?sort=downloads&search=cvt
    image_size = 224
    processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-22k-224")
    model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-22k-224")
    print(model)
    preprocessing = functools.partial(load_preprocess_images, processor=processor, image_size=image_size)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    #wrapper = PytorchWrapper(identifier='model-lecs', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224

    return wrapper


def get_layers(name):
    assert name == 'convnextv2-tiny'
    
    return ['convnextv2.encoder.stages.0.layers.0.dwconv', 'convnextv2.encoder.stages.0.layers.1.dwconv', 'convnextv2.encoder.stages.0.layers.2.dwconv',
            'convnextv2.encoder.stages.1.layers.0.dwconv', 'convnextv2.encoder.stages.1.layers.1.dwconv', 'convnextv2.encoder.stages.1.layers.2.dwconv',
            'convnextv2.encoder.stages.2.layers.0.dwconv', 'convnextv2.encoder.stages.2.layers.1.dwconv', 'convnextv2.encoder.stages.2.layers.2.dwconv',
            'convnextv2.encoder.stages.2.layers.3.dwconv', 'convnextv2.encoder.stages.2.layers.4.dwconv', 'convnextv2.encoder.stages.2.layers.5.dwconv',
            'convnextv2.encoder.stages.2.layers.6.dwconv', 'convnextv2.encoder.stages.2.layers.7.dwconv', 'convnextv2.encoder.stages.2.layers.8.dwconv',
            'convnextv2.encoder.stages.3.layers.0.dwconv', 'convnextv2.encoder.stages.3.layers.1.dwconv', 'convnextv2.encoder.stages.3.layers.2.dwconv',
            ]


def get_bibtex(model_identifier):
    """"""


# def load_preprocess_images(image_filepaths, image_size, processor=None, **kwargs):
#     images = load_images(image_filepaths)
#     # images = [<PIL.Image.Image image mode=RGB size=400x400 at 0x7F8654B2AC10>, ...]
#     images = [image.resize((image_size, image_size)) for image in images]
#     if processor is not None:
#         images = [processor(images=image, return_tensors="pt", **kwargs) for image in images]
#         if len(images[0].keys()) != 1:
#             raise NotImplementedError(f'unknown processor for getting model {processor}')
#         assert list(images[0].keys())[0] == 'pixel_values'
#         images = [image['pixel_values'] for image in images]
#         images = torch.cat(images)
#         images = images.cpu().numpy()
#     else:
#         images = preprocess_images(images, image_size=image_size, **kwargs)
#     return images


# def load_images(image_filepaths):
#     return [load_image(image_filepath) for image_filepath in image_filepaths]


# def load_image(image_filepath):
#     with Image.open(image_filepath) as pil_image:
#         if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper() \
#                 and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
#             # work around to https://github.com/python-pillow/Pillow/issues/1144,
#             # see https://stackoverflow.com/a/30376272/2225200
#             return pil_image.copy()
#         else:  # make sure potential binary images are in RGB
#             rgb_image = Image.new("RGB", pil_image.size)
#             rgb_image.paste(pil_image)
#             return rgb_image


# def preprocess_images(images, image_size, **kwargs):
#     preprocess = torchvision_preprocess_input(image_size, **kwargs)
#     images = [preprocess(image) for image in images]
#     images = np.concatenate(images)
#     return images


# def torchvision_preprocess_input(image_size, **kwargs):
#     from torchvision import transforms
#     return transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         torchvision_preprocess(**kwargs),
#     ])


# def torchvision_preprocess(normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
#     from torchvision import transforms
#     return transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=normalize_mean, std=normalize_std),
#         lambda img: img.unsqueeze(0)
#     ])


# def create_static_video(image, num_frames, normalize_0to1=False, channel_dim=3):
#     '''
#     Create a static video with the same image in all frames.
#     Args:
#         image (PIL.Image.Image): Input image.
#         num_frames (int): Number of frames in the video.
#     Returns:
#         result (np.ndarray): np array of frames of shape (num_frames, height, width, 3).
#     '''
#     frames = []
#     for _ in range(num_frames):
#         frame = np.array(image)
#         if normalize_0to1:
#             frame = frame / 255.
#         if channel_dim == 1:
#             frame = frame.transpose(2, 0, 1)
#         frames.append(frame)
#     return np.stack(frames)


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