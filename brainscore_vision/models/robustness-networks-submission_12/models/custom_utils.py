"""
Change some components of model-tools for the robustness models. 
"""
from model_tools.activations.core import ActivationsExtractorHelper
from collections import OrderedDict

import logging
import numpy as np
from PIL import Image

from model_tools.utils import fullname

class ActivationsExtractorHelperRobustness(ActivationsExtractorHelper):
    def _get_batch_activations(self, inputs, layer_names, batch_size):
        inputs, num_padding = self._pad(inputs, batch_size)
        preprocessed_inputs = self.preprocess(inputs)
        activations = self.get_activations(preprocessed_inputs, layer_names)
        assert isinstance(activations, OrderedDict)
        activations = self._unpad(activations, num_padding)
        return activations

class PytorchWrapperRobustness:
    def __init__(self, model, preprocessing, identifier=None, forward_kwargs=None, *args, **kwargs):
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
        self._forward_kwargs = forward_kwargs or {}

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

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    def get_activations(self, images, layer_names):
        import torch
        from torch.autograd import Variable
        images = [torch.from_numpy(image) if not isinstance(image, torch.Tensor) else image for image in images]
#         images = Variable(torch.stack(images))
        images = torch.stack(images)
        images = images.to(self._device)
        self._model.eval()

        (predictions, _, all_outputs), orig_im = self._model(images, with_latent=True, fake_relu=True)
        layer_results = OrderedDict()
        for layer_name in layer_names:
            if layer_name == 'logits':
                layer_results[layer_name] = all_outputs['final'].detach().cpu().numpy()
#                 layer_results[layer_name] = all_outputs['final']
            else:
                layer_results[layer_name] = all_outputs[layer_name].detach().cpu().numpy()
#                 layer_results[layer_name] = all_outputs[layer_name]

#         layer_results = OrderedDict()
#         hooks = []

#         for layer_name in layer_names:
#             layer = self.get_layer(layer_name)
#             hook = self.register_hook(layer, layer_name, target_dict=layer_results)
#             hooks.append(hook)

#         with torch.no_grad():
#             self._model(images, **self._forward_kwargs)
#         for hook in hooks:
#             hook.remove()
        return layer_results

    def get_layer(self, layer_name):
        raise NotImplementedError('self.get_layer is not applicable for robust models')
#         if layer_name == 'logits':
#             return self._output_layer()
#         module = self._model
#         for part in layer_name.split(SUBMODULE_SEPARATOR):
#             module = module._modules.get(part)
#             assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
#         return module

    def _output_layer(self):
        raise NotImplementedError('self._output_layer is not applicable for robust models')
#         module = self._model
 #        while module._modules:
 #            module = module._modules[next(reversed(module._modules))]
 #        return module

    @classmethod
    def _tensor_to_numpy(cls, output):
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
