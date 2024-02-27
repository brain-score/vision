from collections import OrderedDict

import logging
import numpy as np
from PIL import Image

from brainscore_vision.model_helpers.utils import fullname
from .base import ActivationWrapper

SUBMODULE_SEPARATOR = '.'


class PytorchWrapper(ActivationWrapper):
    def __init__(self, model, preprocessing, forward_kwargs=None):
        import torch
        logger = logging.getLogger(fullname(self))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device {self._device}")
        self._model = model
        self._model = self._model.to(self._device)
        # preprocessing: List[input.Stimulus] -> actual model inputs
        self._preprocess = preprocessing
        self._forward_kwargs = forward_kwargs or {}

    def forward(self, model_inputs):
        return self._model(model_inputs, **self._forward_kwargs)

    def get_activations(self, inputs, layer_names):
        import torch
        inputs = self._preprocess(inputs)
        self._model.eval()

        layer_results = OrderedDict()
        hooks = []

        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            hook = self.register_hook(layer, layer_name, target_dict=layer_results)
            hooks.append(hook)

        with torch.no_grad():
            self.forward(inputs)
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
    def _tensor_to_numpy(cls, output):
        return output.cpu().data.numpy()
    
    @staticmethod
    # rewrite this method to process the activation for different models
    def process_activation(layer, layer_name, input, activation):
        return activation

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name, target_dict=target_dict):
            output = self.process_activation(_layer, name, _input, output)
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
