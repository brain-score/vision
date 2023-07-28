'''
    FeatureExtractor class that allows you to retain outputs of any layer.
    
    This class uses PyTorch's "forward hooks", which let you insert a function
    that takes the input and output of a module as arguements.
    
    In this hook function you can insert tasks like storing the intermediate values,
    or as we'll do in the FeatureEditor class, actually modify the outputs.
    
    Adding these hooks can cause headaches if you don't "remove" them 
    after you are done with them. For this reason, the FeatureExtractor is 
    setup to be used as a context, which sets up the hooks when
    you enter the context, and removes them when you leave:
    
    with FeatureExtractor(model, layer_name) as extractor:
        features = extractor(imgs)
    
    If there's an error in that context (or you cancel the operation),
    the __exit__ function of the feature extractor is executed,
    which we've setup to remove the hooks. This will save you 
    headaches during debugging/development.
    
'''

import torch
import torch.nn as nn
from torchvision import models
from pdb import set_trace

__all__ = ['FeatureExtractor']

class FeatureExtractor(nn.Module):
    def __init__(self, model, layers, detach=True, clone=True, retain=False, device='cpu'):
        layers = [layers] if isinstance(layers, str) else layers
        super().__init__()
        self.model = model
        self.layers = layers
        self.detach = detach
        self.clone = clone
        self.device = device
        self.retain = retain
        self._features = {layer: torch.empty(0) for layer in layers}        
        self.hooks = {}
        
    def hook_layers(self):        
        self.remove_hooks()
        for layer_id in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            self.hooks[layer_id] = layer.register_forward_hook(self.save_outputs_hook(layer_id))
    
    def remove_hooks(self):
        for layer_id in self.layers:
            if self.retain==False:
                self._features[layer_id] = torch.empty(0)
            if layer_id in self.hooks:
                self.hooks[layer_id].remove()                
                del self.hooks[layer_id]
    
    def __enter__(self, *args): 
        self.hook_layers()
        return self
    
    def __exit__(self, *args): 
        self.remove_hooks()
        
    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            if self.detach: output = output.detach()
            if self.clone: output = output.clone()
            if self.device: output = output.to(self.device)
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features
    
def get_layers(model, parent_name='', layer_info=[]):
    for module_name, module in model.named_children():
        layer_name = parent_name + '.' + module_name
        if len(list(module.named_children())):
            layer_info = get_layers(module, layer_name, layer_info=layer_info)
        else:
            layer_info.append(layer_name.strip('.'))
    
    return layer_info

def get_layer(m, layers):
    layer = layers.pop(0)
    m = getattr(m, layer)
    if len(layers) > 0:
        return get_layer(m, layers)
    return m

def get_layer_type(model, layer_name):
    m = get_layer(model, layer_name.split("."))
    return m.__class__.__name__
            
def convert_relu_layers(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU) and child.inplace==True:
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu_layers(child)