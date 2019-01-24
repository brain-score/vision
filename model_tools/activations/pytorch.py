import logging
from collections import OrderedDict

import numpy as np
from PIL import Image

from model_tools.activations.core import ActivationsExtractorHelper
from model_tools.utils import fullname

_logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)

SUBMODULE_SEPARATOR = '.'


class PytorchWrapper:
    def __init__(self, model, preprocessing, identifier=None, *args, **kwargs):
        import torch
        self._logger = logging.getLogger(fullname(self))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._logger.debug(f"Using device {self._device}")
        self._model = model
        self._model = self._model.to(self._device)
        identifier = identifier or model.__module__
        self._extractor = ActivationsExtractorHelper(
            identifier=identifier, get_activations=self.get_activations, preprocessing=preprocessing,
            *args, **kwargs)
        self.from_stimulus_set = self._extractor.from_stimulus_set
        self.from_paths = self._extractor.from_paths

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    def get_activations(self, images, layer_names):
        import torch
        from torch.autograd import Variable
        images = [torch.from_numpy(image) for image in images]
        images = Variable(torch.stack(images))
        images = images.to(self._device)

        layer_results = OrderedDict()
        hooks = []

        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            hook = self.register_hook(layer, layer_name, target_dict=layer_results)
            hooks.append(hook)

        self._model.eval()
        self._model(images)
        for hook in hooks:
            hook.remove()
        return layer_results

    def get_layer(self, layer_name):
        module = self._model
        for part in layer_name.split(SUBMODULE_SEPARATOR):
            module = module._modules.get(part)
            assert module is not None, "No submodule found for layer {}, at part {}".format(layer_name, part)
        return module

    def store_layer_output(self, layer_results, layer_name, output):
        layer_results[layer_name] = output.cpu().data.numpy()

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            self.store_layer_output(target_dict, name, output)

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


def load_preprocess_images(image_filepaths, image_size):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = preprocess_images(images, image_size=image_size)
    return images


def load_image(image_filepath):
    with Image.open(image_filepath) as image:
        if 'L' not in image.mode.upper() and 'A' not in image.mode.upper():  # not binary and not alpha
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", image.size)
            rgb_image.paste(image)
            return rgb_image


def preprocess_images(images, image_size):
    images = [preprocess_image(image, image_size) for image in images]
    images = np.concatenate(images)
    return images


def preprocess_image(image, image_size):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image * 255))
    image = torchvision_preprocess_input(image_size)(image)
    return image


def torchvision_preprocess_input(image_size, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])
