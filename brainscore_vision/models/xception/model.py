import functools
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import timm
# import torch
import torch.nn as nn
# from torch.nn import functional as F


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'xception'
    model = timm.create_model('xception', pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=299)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing, batch_size=4)
    wrapper.image_size = 299
    return wrapper


def get_layers(name):
    assert name == 'xception'
    # model = timm.create_model('xception', pretrained=True)
    # for n, _ in model.named_modules():
    #     print(n)
    layer_names = (
        # Block 1 (2 layers)
        # [f'block1.rep.{i}.pointwise' for i in [0, 3]] +
        # Block 2 (1 layer)
        # ['block2.rep.4.pointwise'] +
        # Block 3 (2 layers)
        [f'block3.rep.{i}.pointwise' for i in [1, 4]] +
        # Block 4 (2 layers)
        [f'block4.rep.{i}.pointwise' for i in [1, 4]] +
        # Blocks 5-11 (3 layers each)
        [f'block{block}.rep.{layer}.pointwise'
         for block in range(5, 12)
         for layer in [1, 4, 7]] +
        # Block 12 (2 layers)
        [f'block12.rep.{i}.pointwise' for i in [1, 4]] +
        # Final layers
        ['conv3.pointwise', 'conv4.pointwise', 'global_pool.pool']
    )
    return layer_names


def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
@article{DBLP:journals/corr/ZagoruykoK16,
@misc{chollet2017xception,
      title={Xception: Deep Learning with Depthwise Separable Convolutions}, 
      author={François Chollet},
      year={2017},
      eprint={1610.02357},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)


class PooledModelWrapper(nn.Module):
#     def __init__(self, base_model, pooling_size=(16, 16)):
#         super().__init__()
#         self.model = base_model
#         self.pooling_size = pooling_size
#         self.hooks = []
#         self.activations = {}
#
#         # Register hooks for each layer
#         for name, _ in base_model.named_modules():
#             layer = self._get_layer(name)
#             hook = layer.register_forward_hook(self._get_hook(name))
#             self.hooks.append(hook)
#
#     def _get_layer(self, name):
#         # Navigate nested model structure using name parts
#         parts = name.split('.')
#         current = self.model
#         for part in parts:
#             current = getattr(current, part)
#         return current
#
#     def _get_hook(self, name):
#         def hook(module, input, output):
#             # Print original dimensions
#             if isinstance(output, torch.Tensor):
#                 print(f"\nLayer {name}")
#                 print(f"Original dimensions: {output.shape}")
#                 # [batch_size, channels, height, width]
#
#             # Apply adaptive pooling to reduce dimensions
#             # Skip pooling for global pool layer which is already pooled
#             if 'global_pool' not in name:
#                 if isinstance(output, torch.Tensor):
#                     # For single tensor output
#                     output = F.adaptive_avg_pool2d(output, self.pooling_size)
#                     print(f"After pooling: {output.shape}")
#                 else:
#                     # For tuple/list outputs, apply to each tensor
#                     output = tuple(F.adaptive_avg_pool2d(o, self.pooling_size)
#                                    if isinstance(o, torch.Tensor) else o
#                                    for o in output)
#             self.activations[name] = output
#
#         return hook
#
#     def forward(self, x):
#         self.activations.clear()
#         return self.model(x)
#
#
# def get_model(name):
#     """
#     Get a model with pooled activations to prevent memory overflow in PCA.
#     """
#     assert name == 'xception'
#     base_model = timm.create_model('xception', pretrained=True)
#     layer_names = get_layers(name)
#
#     # Create wrapped model with pooling
#     pooled_model = PooledModelWrapper(base_model, pooling_size=(16, 16))
#
#     preprocessing = functools.partial(load_preprocess_images, image_size=299)
#     wrapper = PytorchWrapper(identifier=name, model=pooled_model,
#                              preprocessing=preprocessing, batch_size=4)
#     wrapper.image_size = 299
#     return wrapper