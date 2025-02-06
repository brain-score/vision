from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.

import torch.nn as nn
from collections import OrderedDict
import inspect

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
class alexnet_backbone(nn.Module):
    def __init__(self, in_channel=3, out_dim=(6,6), w=1):
        super(alexnet_backbone, self).__init__()
        
        self.num_features = int(256*w) * out_dim[0] * out_dim[1]
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, int(96*w), 11, 4, 2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(int(96*w), int(256*w), 5, 1, 2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(int(256*w), int(384*w), 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(int(384*w), int(384*w), 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(int(384*w), int(256*w), 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(out_dim)
        self.head = nn.Identity() # add task-specific head 
        
    def forward_features(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.avgpool(x)
        return x
    
    def forward_head(self, x, return_layer_outputs=False):
        if hasattr(self, 'head') and 'return_layer_outputs' in inspect.signature(self.head.forward).parameters:
            x = self.head(x, return_layer_outputs=return_layer_outputs)
        elif hasattr(self, 'head'):
            x = self.head(x)
        return x
    
    def forward(self, x, return_layer_outputs=True):
        x = self.forward_features(x)
        x = self.forward_head(x, return_layer_outputs=return_layer_outputs)
        return x
    
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    
class MLP(nn.Module):
    def __init__(self, mlp_spec, proj_relu=False, mlp_coeff=1, output_bias=False, 
                 norm=nn.BatchNorm1d, nonlin=nn.ReLU, l2norm=False, dropout=None, dropout_first=False):
        super(MLP, self).__init__()

        # Construct the MLP layers as before
        self.layers = self._construct_mlp_layers(mlp_spec, proj_relu, mlp_coeff, output_bias, norm, nonlin, l2norm, dropout, dropout_first)

    def _construct_mlp_layers(self, mlp_spec, proj_relu, mlp_coeff, output_bias, norm, nonlin, l2norm, dropout, dropout_first):
        # check whether activation function takes the inplace parameter
        has_inplace = 'inplace' in inspect.signature(nonlin.__init__).parameters
        dropout_last = ~dropout_first
        
        # This method constructs the MLP layers
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        f[-2] = int(f[-2] * mlp_coeff)
        
        # constuct each linear block
        for i in range(len(f) - 2):
            fc_layers = []
            
            if dropout is not None and dropout_first:
                fc_layers += [('dropout', nn.Dropout(dropout, inplace=True))]
                        
            fc_layers += [('linear', nn.Linear(f[i], f[i + 1]))]
            
            if norm is not None:
                fc_layers += [('norm', norm(f[i + 1]))]
            
            if nonlin is not None:
                fc_layers += [ ('nonlin', nonlin(inplace=True) if has_inplace else nonlin())]                                               
            
            if dropout is not None and dropout_last:
                fc_layers += [('dropout', nn.Dropout(dropout, inplace=False))]
                
            layers.append(nn.Sequential(OrderedDict(fc_layers)))
        
        # get layer of last linear block
        last_layers = [
            ('linear', nn.Linear(f[-2], f[-1], bias=output_bias))
        ]
        
        if proj_relu:
            last_layers += [('nonlin', nn.ReLU(True))]
            
        layers.append(nn.Sequential(OrderedDict(last_layers)))
        
        if l2norm:
            layers.append(Normalize(power=2))
            
        return nn.ModuleList(layers)

    def forward(self, x, return_layer_outputs=True):
        x = x.flatten(start_dim=1)
        list_outputs = [x.detach()]
        for layer in self.layers:
            x = layer(x)
            list_outputs.append(x.detach())  # Store the output after each layer

        # The final value of x is the embedding
        embeddings = x
        
        if return_layer_outputs:
            return embeddings, list_outputs
        
        return embeddings    

class LinearProbes(nn.Module):
    def __init__(self, mlp_spec, mlp_coeff=1, num_classes=1000):
        super().__init__()
        print("LINEAR PROBE NUM CLASSES:", num_classes)
        f = list(map(int, mlp_spec.split("-")))
        f[-2] = int(f[-2] * mlp_coeff)
        self.probes = []
        for num_features in f:
            self.probes.append(nn.Linear(num_features, num_classes))
        self.probes = nn.Sequential(*self.probes)

    def forward(self, list_outputs, binary=False):
        return [self.probes[i](list_outputs[i]) for i in range(len(list_outputs))]
    
class SSLModelWithLinearProbes(nn.Module):
    def __init__(self, model, probes, probe_layer=0, img_size=224):
        super(SSLModelWithLinearProbes, self).__init__()
        self.model = model
        self.probes = probes
        self.probe_layer = probe_layer

    def forward(self, x, probe_layer=None):
        probe_layer = self.probe_layer if probe_layer is None else probe_layer
        embeddings, list_outputs = self.model(x)
        logits = [self.probes[layer](list_outputs[layer]) for layer in range(len(self.probes))]        
        logits = logits[probe_layer]
        return logits

def load_checkpoint(model, url):
    checkpoint = load_state_dict_from_url(url, check_hash=True, map_location='cpu')
    state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
    state_dict = {k.replace("base_arch.",""):v for k,v in state_dict.items()}
    del state_dict['loss_fun.temperature']
    probes = {k.replace("module.probes.",""):v for k,v in checkpoint['probes'].items()}

    msg = model.model.load_state_dict(state_dict)
    print(f"model: {msg}")
    msg = model.probes.load_state_dict(probes)
    print(f"probes: {msg}")
    
    return model

def alexnet_mlp(probe_layer, in_channel=3, w=1, mlp='4096-4096-4096', fc_num_classes=1000, out_dim=(6,6),
                output_bias=False, dropout=None):
    network = alexnet_backbone(in_channel=in_channel, out_dim=out_dim, w=w)
    mlp = f'{network.num_features}-{mlp}'
    network.head = MLP(mlp, proj_relu=False, mlp_coeff=1, output_bias=output_bias,
                       norm=nn.BatchNorm1d, nonlin=nn.ReLU, l2norm=False, 
                       dropout=dropout, dropout_first=False)
    probes = LinearProbes(mlp, num_classes=fc_num_classes).probes
    
    model = SSLModelWithLinearProbes(network, probes, probe_layer=probe_layer)
    
    return model

def alexnet_w1_mlp4096_simclr_baseline_64715_probe0():
    url = 'https://visionlab-members.s3.wasabisys.com/alvarez/Projects/configural_shape_private/baseline_models/in1k/alexnet_w1_mlp/simclr/20240513_182357/final_weights-647159ec62.pth'
    model = alexnet_mlp(probe_layer=0)
    model = load_checkpoint(model, url)
    
    return model

def alexnet_w1_mlp4096_simclr_baseline_64715_probe1():
    url = 'https://visionlab-members.s3.wasabisys.com/alvarez/Projects/configural_shape_private/baseline_models/in1k/alexnet_w1_mlp/simclr/20240513_182357/final_weights-647159ec62.pth'
    model = alexnet_mlp(probe_layer=1)
    model = load_checkpoint(model, url)
    
    return model

def alexnet_w1_mlp4096_simclr_ratio1augs_d3fbd_probe0():
    url = 'https://visionlab-members.s3.wasabisys.com/alvarez/Projects/configural_shape_private/simclr_correlated_crops/in1k/alexnet_w1_mlp/simclr/20240513_211743/final_weights-d3fbd988f5.pth'
    model = alexnet_mlp(probe_layer=0)
    model = load_checkpoint(model, url)
    return model

def alexnet_w1_mlp4096_simclr_ratio1augs_d3fbd_probe1():
    url = 'https://visionlab-members.s3.wasabisys.com/alvarez/Projects/configural_shape_private/simclr_correlated_crops/in1k/alexnet_w1_mlp/simclr/20240513_211743/final_weights-d3fbd988f5.pth'
    model = alexnet_mlp(probe_layer=1)
    model = load_checkpoint(model, url)
    return model

def get_model_list():
    return ['alexnet_w1_mlp4096_simclr_baseline_64715_probe0', 'alexnet_w1_mlp4096_simclr_ratio1augs_d3fbd_probe0']

def get_model(name):
    assert name in get_model_list()
    preprocessing = functools.partial(load_preprocess_images, image_size=224, 
                                      normalize_mean=(0.485, 0.456, 0.406), 
                                      normalize_std=(0.229, 0.224, 0.225))
    
    if name=='alexnet_w1_mlp4096_simclr_baseline_64715_probe0':
        model = alexnet_w1_mlp4096_simclr_baseline_64715_probe0()
    elif name=='alexnet_w1_mlp4096_simclr_ratio1augs_d3fbd_probe0':
        model = alexnet_w1_mlp4096_simclr_ratio1augs_d3fbd_probe0()
        
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name in get_model_list()
    return ['model.conv_block_1', 'model.conv_block_2', 'model.conv_block_3',
            'model.conv_block_4', 'model.conv_block_5', 'model.head.layers.0',
            'model.head.layers.1', 'model.head.layers.2', 
            'probes.0', 'probes.1', 'probes.2', 'probes.3']


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
