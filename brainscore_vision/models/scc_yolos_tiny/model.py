from brainscore_vision.model_helpers.check_submission import check_models
import functools
import numpy as np
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image

class Yolos_Tiny_PytorchWrapper(PytorchWrapper):
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)  # retrieve original output
        if 'logits' in kwargs.get('layers', []):
            result = result.isel(neuroid=slice(1, None))  # remove background class in last layer
        return result
    @staticmethod
    def _tensor_to_numpy(output):
        if isinstance(output, tuple):
            output = output[0]
        return output.cpu().data.numpy()

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if isinstance(output, tuple):
                output = output[0]
            target_dict[name] = Yolos_Tiny_PytorchWrapper._tensor_to_numpy(output)
        hook = layer.register_forward_hook(hook_function)
        return hook

def get_model_list():
    return ['yolos_tiny']

def get_model(name):
    assert name == 'yolos_tiny'
    image_size = 224  
    processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    preprocessing = functools.partial(load_preprocess_images, processor=processor, image_size=image_size)
    wrapper = Yolos_Tiny_PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper

def get_layers(name):
    assert name == "yolos_tiny"
    layer_names = [
        "vit.encoder.layer.0.attention.output.dense",
        "vit.encoder.layer.0.output.dense",
        "vit.encoder.layer.2.attention.output.dense",
        "vit.encoder.layer.2.output.dense",
        "vit.encoder.layer.4.attention.output.dense",
        "vit.encoder.layer.4.output.dense",
        "vit.encoder.layer.6.attention.output.dense",
        "vit.encoder.layer.6.output.dense",
        "vit.encoder.layer.8.attention.output.dense",
        "vit.encoder.layer.8.output.dense",
        "vit.encoder.layer.10.attention.output.dense",
        "vit.encoder.layer.10.output.dense",
        "vit.encoder.layer.11.attention.output.dense",
        "vit.encoder.layer.11.output.dense",

        "vit.layernorm",

        "class_labels_classifier.layers.0",
        "class_labels_classifier.layers.1",
        "class_labels_classifier.layers.2",
    ]
    return layer_names

def get_bibtex(model_identifier):
    return """@article{fang2021yolos,
      title={You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection},
      author={Tianyang Fang and Wen Wang and Bohan Zhuang and Chunhua Shen},
      journal={arXiv preprint arXiv:2106.00666},
      year={2021}
    }"""

def load_preprocess_images(image_filepaths, image_size, processor=None, **kwargs):
    images = [Image.open(filepath).convert('RGB') for filepath in image_filepaths]
    images = [image.resize((image_size, image_size)) for image in images]
    if processor is not None:
        images = [processor(images=image, return_tensors="pt", **kwargs)['pixel_values'] for image in images]
        images = torch.cat(images).cpu().numpy()
    return images

if __name__ == '__main__':
    check_models.check_base_models(__name__)
