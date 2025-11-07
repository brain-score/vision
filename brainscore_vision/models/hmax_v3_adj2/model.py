from brainscore_vision.model_helpers.check_submission import check_models
import functools
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from io import BytesIO
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images


from brainscore_vision.models.hmax_v3_adj2.local_timm.models.RESMAX import hmax_v3_adj

def load_model(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_url = "https://huggingface.co/cmulliken/hmax_v3_adj/resolve/main/model_best.pth.tar"
    args_url = "https://huggingface.co/cmulliken/hmax_v3_adj/resolve/main/args.yaml"

    # Download checkpoint into memory
    response = requests.get(checkpoint_url, stream=True)
    response.raise_for_status()
    checkpoint_file = BytesIO(response.content)

    summmary_response = requests.get(args_url, stream=True)
    summmary_response.raise_for_status()
    args_file = BytesIO(summmary_response.content)

    
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    args_dict = yaml.safe_load(args_file)

    kwargs = {
        'ip_scale_bands': args_dict['model_kwargs']['ip_scale_bands'],
        'classifier_input_size': args_dict['model_kwargs']['classifier_input_size'],
        'bypass': args_dict['model_kwargs']['bypass'],
    }

    model = hmax_v3_adj(
        pretrained=False,
        **kwargs
    )
    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

class ListToTensorWrapper(nn.Module):
    """Wraps a module that returns lists, converting to tensors for external hooks"""
    def __init__(self, wrapped_module):
        super().__init__()
        self.wrapped = wrapped_module

    def forward(self, *args, **kwargs):
        out = self.wrapped(*args, **kwargs)
        # Always pass through lists internally - hooks will see the list
        # BrainScore will handle the conversion when it extracts activations
        return out

class HMAXActivationWrapper(PytorchWrapper):
    """
    Custom PytorchWrapper that handles list outputs from HMAX multi-scale layers.

    HMAX layers output lists of tensors (one per scale), but BrainScore expects
    single tensors. This wrapper intercepts activation extraction and converts
    lists to tensors without modifying the model's forward pass.
    """

    def get_activations(self, images, layer_names):
        """Override to handle list outputs from HMAX layers"""
        # Temporarily add hooks that convert lists to tensors
        hooks = []

        def list_to_tensor_hook(module, input, output):
            if isinstance(output, list) and len(output) > 0:
                # Average across scales
                try:
                    pooled = [F.adaptive_avg_pool2d(o, output[0].shape[-2:]) for o in output]
                    return torch.stack(pooled, dim=0).mean(dim=0)
                except:
                    return output[0]  # Fallback to first scale
            return output

        # Apply hooks only to requested layers during activation extraction
        for layer_name in layer_names:
            try:
                layer = self.get_layer(layer_name)
                hook = layer.register_forward_hook(list_to_tensor_hook)
                hooks.append(hook)
            except:
                pass  # Layer might not exist or not be hookable

        try:
            # Get activations with hooks active
            activations = super().get_activations(images, layer_names)
            return activations
        finally:
            # Always remove hooks to restore normal operation
            for hook in hooks:
                hook.remove()

def get_model(name):
    assert name == 'hmax_v3_adj2'
    base_model = load_model()

    # --- define wrapper that only affects what Brain-Score sees ---
    class BrainscoreSafeModel(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            y = self.inner(x)
            # if Brain-Score hooks at the top model level, make sure it sees a tensor
            if isinstance(y, tuple):
                # Model returns (output, loss) during training or eval
                return y[0] if len(y) == 2 else y
            if isinstance(y, list):
                pooled = [F.adaptive_avg_pool2d(o, y[0].shape[-2:]) for o in y]
                y = torch.stack(pooled, dim=0).mean(dim=0)
            return y

        # crucial: expose submodules unchanged so Brain-Score can hook them
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    model = BrainscoreSafeModel(base_model)

    preprocessing = functools.partial(load_preprocess_images, image_size=322)
    wrapper = HMAXActivationWrapper(identifier='hmax_v3_adj2', model=model, preprocessing=preprocessing)
    wrapper.image_size = 322
    return wrapper

def get_layers(name):
    assert name == 'hmax_v3_adj2'
    return ['inner.model_backbone.s1', 'inner.model_backbone.c1', 'inner.model_backbone.s2', 'inner.model_backbone.c2', 'inner.model_backbone.s2b', 'inner.model_backbone.c2b_score', 'inner.model_backbone.s3']

def get_bibtex(model_identifier):
    return """"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)
