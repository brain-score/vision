import os
import logging
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper
from brainscore_vision.model_helpers.utils import fullname

SUBMODULE_SEPARATOR = '.'


class PytorchWrapper:
    def __init__(self, model, preprocessing, identifier=None, *args, **kwargs):
        import torch
        logger = logging.getLogger(fullname(self))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device {self._device}")
        self._model = model
        self._model = self._model.to(self._device)
        identifier = identifier or model.__class__.__name__
        self._extractor = self._build_extractor(
            identifier=identifier, preprocessing=preprocessing, get_activations=self.get_activations, *args, **kwargs)
        self._extractor.insert_attrs(self)

    def _build_extractor(self, identifier, preprocessing, get_activations, *args, **kwargs):
        return ActivationsExtractorHelper(
            identifier=identifier, get_activations=get_activations, preprocessing=preprocessing,
            *args, **kwargs)

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    def __call__(self, *args, **kwargs):
        previous_value = os.getenv('RESULTCACHING_DISABLE', '')
        os.environ['RESULTCACHING_DISABLE'] = 'model_tools.activations'
        result = self._extractor(*args, **kwargs)
        os.environ['RESULTCACHING_DISABLE'] = previous_value
        return result

    def get_activations(self, images, layer_names):
        import torch
        from torch.autograd import Variable
        images = [torch.from_numpy(image) for image in images]
        images = Variable(torch.stack(images))
        images = images.to(self._device)
        self._model.eval()

        layer_results = OrderedDict()
        hooks = []

        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            hook = self.register_hook(layer, layer_name, target_dict=layer_results)
            hooks.append(hook)

        self._model(images)
        for hook in hooks:
            hook.remove()
        return layer_results

    def get_layer(self, layer_name):
        if layer_name == 'logits':
            return self._output_layer()
        module = self._model
        for part in layer_name.split(SUBMODULE_SEPARATOR):
            module = module._modules.get(part)
            assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
        return module

    def _output_layer(self):
        module = self._model
        while module._modules:
            module = module._modules[next(reversed(module._modules))]
        return module

    @classmethod
    def set_weights(cls, output, shape):
        if len(shape) == 5:
            return torch.cat([out[:shape[0],None, :shape[1], :shape[2], :shape[3], :shape[4]] for out in output], 1)
        if len(shape) == 4:
            return torch.cat([out[:shape[0],None, :shape[1], :shape[2], :shape[3]] for out in output], 1)
        elif len(shape) == 3:
            return torch.cat([out[:shape[0],None,:shape[1], :shape[2]] for out in output], 1)
        elif len(shape) == 2:
            return torch.cat([out[:shape[0],None ,:shape[1]] for out in output], 1)
        elif len(shape) == 1:
            return torch.cat([out[:shape[0], None] for out in output], 1)
        else:
            raise ValueError(f"Unexpected shape {shape}")

    @classmethod
    def _tensor_to_numpy(cls, output):
        if isinstance(output, list):
            zero = output[0]
            if isinstance(zero, list):
                length = len(zero[0].shape)
                shape = np.full(length, 100000)
                for li in output:
                    for item in li:
                        for j in range(len(item.shape)):
                            if shape[j] > item.shape[j]:
                                shape[j] = item.shape[j]
                shape = shape.astype(int)
                new_weights = []
                for i in range(len(output)):
                    outer = output[i]
                    new_weights.append(cls.set_weights(outer, shape))
                new_weights = cls.set_weights(new_weights, new_weights[0].shape)
                if len(new_weights.shape) == 6:
                    new_weights = new_weights.reshape(new_weights.shape[0],new_weights.shape[1]*new_weights.shape[2]*new_weights.shape[3],new_weights.shape[4],new_weights.shape[5])
                elif len(new_weights.shape) == 5:
                    new_weights = new_weights.reshape(new_weights.shape[0],new_weights.shape[1]*new_weights.shape[2],new_weights.shape[3],new_weights.shape[4])
                elif len(new_weights.shape) == 4:
                    pass  # ok already
                elif len(new_weights.shape) == 3:
                    new_weights = new_weights.reshape(new_weights.shape[0],1,new_weights.shape[1],new_weights.shape[2])
                elif len(new_weights.shape) == 2:
                    new_weights = new_weights.reshape(new_weights.shape[0],1,1,new_weights.shape[1])
                else:
                    raise ValueError(f"Unexpected shape {new_weights.shape}")
                return new_weights
            else:
                length = len(zero.shape)
                shape = np.full(length, 10000)
                for item in output:
                    for j in range(len(item.shape)):
                        if shape[j] > item.shape[j]:
                            shape[j] = item.shape[j]
                shape = shape.astype(int)
                new_weights = cls.set_weights(output, shape)
                if len(new_weights.shape) > 4:
                    new_weights = new_weights.reshape(new_weights.shape[0],new_weights.shape[1]*new_weights.shape[2],new_weights.shape[3],new_weights.shape[4])
                if len(new_weights.shape) == 3:
                    new_weights = new_weights.reshape(new_weights.shape[0],1,new_weights.shape[1],new_weights.shape[2])
                if len(new_weights.shape) == 2:
                    new_weights = new_weights.reshape(new_weights.shape[0],1,1,new_weights.shape[1])
            return new_weights
        return output.cpu().data.numpy()

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            target_dict[name] = PytorchWrapper._tensor_to_numpy(output)

        hook = layer.register_forward_hook(hook_function)
        return hook

    def __repr__(self):
        return repr(self._model)

    def layers(self):
        for name, module in self._model.named_modules():
            if len(list(module.children())) > 0:  # this module only holds other modules
                continue
            yield name, module

    def graph(self):
        import networkx as nx
        g = nx.DiGraph()
        for layer_name, layer in self.layers():
            g.add_node(layer_name, object=layer, type=type(layer))
        return g


def load_preprocess_images(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images


def load_images(image_filepaths):
    return [load_image(image_filepath) for image_filepath in image_filepaths]


def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper():  # not binary and not alpha
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

