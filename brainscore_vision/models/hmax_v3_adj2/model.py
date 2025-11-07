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
    single tensors. This wrapper post-processes activations after extraction,
    converting lists to tensors without interfering with the model's forward pass.
    """

    def get_activations(self, images, layer_names):
        """Override to handle list outputs from HMAX layers"""
        print(f"\n=== HMAX get_activations called ===")
        print(f"Requested layers: {layer_names}")
        print(f"Number of images: {len(images)}")

        # Get activations normally (BrainScore's hooks will capture lists)
        activations_dict = super().get_activations(images, layer_names)

        print(f"\nActivations retrieved from parent class:")
        for layer_name, activation in activations_dict.items():
            print(f"  {layer_name}: type={type(activation).__name__}", end="")
            if isinstance(activation, list):
                print(f", list_length={len(activation)}", end="")
                if len(activation) > 0:
                    print(f", first_element_shape={activation[0].shape if hasattr(activation[0], 'shape') else 'N/A'}")
                else:
                    print()
            elif hasattr(activation, 'shape'):
                print(f", shape={activation.shape}")
            else:
                print()

        # Post-process: convert any list values to tensors
        print(f"\nPost-processing activations...")
        for layer_name, activation in activations_dict.items():
            if isinstance(activation, list) and len(activation) > 0:
                print(f"  {layer_name}: Converting list to tensor...")
                # Convert list of tensors to single tensor by averaging across scales
                try:
                    print(f"    List has {len(activation)} scales")
                    for i, a in enumerate(activation):
                        print(f"      Scale {i}: shape={a.shape if hasattr(a, 'shape') else type(a)}")

                    # Ensure all tensors have the same spatial dimensions
                    pooled = [F.adaptive_avg_pool2d(a, activation[0].shape[-2:]) for a in activation]
                    # Stack and average
                    tensor_activation = torch.stack(pooled, dim=0).mean(dim=0)
                    activations_dict[layer_name] = tensor_activation
                    print(f"    ✓ Converted to tensor with shape: {tensor_activation.shape}")
                except Exception as e:
                    # If conversion fails, use the first scale
                    print(f"    ✗ Conversion failed: {e}")
                    print(f"    Using first scale only")
                    activations_dict[layer_name] = activation[0]
            else:
                print(f"  {layer_name}: Already a tensor, no conversion needed")

        print(f"\nFinal activations:")
        for layer_name, activation in activations_dict.items():
            act_type = type(activation).__name__
            act_shape = activation.shape if hasattr(activation, 'shape') else 'N/A'
            print(f"  {layer_name}: type={act_type}, shape={act_shape}")

        print(f"=== HMAX get_activations completed ===\n")
        return activations_dict

def get_model(name):
    print(f"\n{'='*80}")
    print(f"HMAX MODEL INITIALIZATION: {name}")
    print(f"{'='*80}")
    assert name == 'hmax_v3_adj2'
    base_model = load_model()
    print(f"Base model loaded: {type(base_model).__name__}")

    # --- define wrapper that only affects what Brain-Score sees ---
    class BrainscoreSafeModel(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            print(f"\n[BrainscoreSafeModel.forward] Input shape: {x.shape}")
            y = self.inner(x)
            print(f"[BrainscoreSafeModel.forward] Inner model output type: {type(y).__name__}")

            # if Brain-Score hooks at the top model level, make sure it sees a tensor
            if isinstance(y, tuple):
                # Model returns (output, loss) during training or eval
                print(f"[BrainscoreSafeModel.forward] Output is tuple of length {len(y)}")
                result = y[0] if len(y) == 2 else y
                print(f"[BrainscoreSafeModel.forward] Returning element 0 with shape: {result.shape if hasattr(result, 'shape') else type(result)}")
                return result
            if isinstance(y, list):
                print(f"[BrainscoreSafeModel.forward] Output is list of {len(y)} elements")
                pooled = [F.adaptive_avg_pool2d(o, y[0].shape[-2:]) for o in y]
                y = torch.stack(pooled, dim=0).mean(dim=0)
                print(f"[BrainscoreSafeModel.forward] Converted list to tensor with shape: {y.shape}")
            print(f"[BrainscoreSafeModel.forward] Returning: {y.shape if hasattr(y, 'shape') else type(y)}\n")
            return y

        # crucial: expose submodules unchanged so Brain-Score can hook them
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    model = BrainscoreSafeModel(base_model)
    print(f"Wrapped model created: {type(model).__name__}")

    preprocessing = functools.partial(load_preprocess_images, image_size=322)
    wrapper = HMAXActivationWrapper(identifier='hmax_v3_adj2', model=model, preprocessing=preprocessing)
    wrapper.image_size = 322
    print(f"Final wrapper created: {type(wrapper).__name__}")
    print(f"Layers that will be exposed: {get_layers('hmax_v3_adj2')}")
    print(f"{'='*80}\n")
    return wrapper

def get_layers(name):
    assert name == 'hmax_v3_adj2'
    return ['inner.model_backbone.s1', 'inner.model_backbone.c1', 'inner.model_backbone.s2', 'inner.model_backbone.c2', 'inner.model_backbone.s2b', 'inner.model_backbone.c2b_score', 'inner.model_backbone.s3']

def get_bibtex(model_identifier):
    return """"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)
