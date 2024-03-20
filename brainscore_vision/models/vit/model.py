import functools

import torchvision.models

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images, load_image

from transformers import ViTImageProcessor, ViTModel

import torch
from torch import nn

BIBTEX = """@misc{wu2020visual,
      title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, 
      author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},
      year={2020},
      eprint={2006.03677},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}"""

LAYERS = []
for i in range(12): # 12
    LAYERS.append(f'model.encoder.layer.{i}')
LAYERS.append('model.encoder')
LAYERS.append('model.layernorm')
LAYERS.append('last_hidden_state')

class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_hidden_state = nn.Identity()
    def forward(self, x):
        # print(type(x))
        # print(x.shape)
        # x = {'pixel_values': x}
        x = self.model(pixel_values=x)
        x = x['last_hidden_state']
        x = self.last_hidden_state(x)
        return x

def get_model():
    # get model
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    custom_wrapper = Wrapper(model)

    # get preprocessor
    image_size = 224
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    preprocessing = functools.partial(load_preprocess_images_torch, processor=processor, image_size=image_size)

    # get brainscore wrapper
    wrapper = PytorchWrapperV2(identifier='vit-base-patch16-224-in21k_debug1118', model=custom_wrapper, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper

def load_preprocess_images_torch(image_filepaths, processor, image_size, **kwargs):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = [image.resize((image_size, image_size)) for image in images]
    images = [processor(images=image, return_tensors="pt", **kwargs) for image in images]
    images = [image['pixel_values'] for image in images]
    images = torch.cat(images)
    images = images.cpu()
    return images

class PytorchWrapperV2(PytorchWrapper):
    def __init__(self, identifier, model, preprocessing):
        super().__init__(identifier, model, preprocessing)

    @classmethod
    def _tensor_to_numpy(cls, output):
        try:
            print("Output shape:", output.shape)
            return output.cpu().data.numpy()
        except:
            print("Output shape:", output[0].shape)
            return output[0].cpu().data.numpy()