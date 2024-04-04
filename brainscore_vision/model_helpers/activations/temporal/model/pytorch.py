from collections import OrderedDict
from typing import Callable, List, Any

import logging

from brainscore_vision.model_helpers.utils import fullname
from .base import ActivationWrapper
from ..inputs import Stimulus


SUBMODULE_SEPARATOR = '.'


def default_process_activation(layer, layer_name, inputs, output):
    # print(layer_name, output.shape)
    return output

class PytorchWrapper(ActivationWrapper):
    def __init__(
            self, 
            identifier : str, 
            model, 
            preprocessing : Callable[[List[Stimulus]], Any], 
            process_output = None, 
            *args, 
            **kwargs
        ):
        import torch
        logger = logging.getLogger(fullname(self))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device {self._device}")
        self._model = model.to(self._device)
        # preprocessing: input.Stimulus -> actual model inputs
        self._preprocess = preprocessing
        self._process_activation = default_process_activation if process_output is None else process_output
        super().__init__(identifier, preprocessing, *args, **kwargs)

    def forward(self, inputs):
        import torch
        tensor = torch.stack(inputs)
        tensor = tensor.to(self._device)
        return self._model(tensor)

    def get_activations(self, inputs, layer_names):
        import torch
        self._model.eval()
        layer_results = OrderedDict()
        hooks = []

        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            hook = self._register_hook(layer, layer_name, target_dict=layer_results)
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

    def _register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name, target_dict=target_dict):
            output = self._process_activation(_layer, name, _input, output)
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
