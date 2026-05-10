"""Brain-Score wrapper for DeepEISNN EI-SNN CIFAR10 ResNet18."""

from __future__ import annotations

import functools
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if not hasattr(np, "int"):
    np.int = np.int32

import torch
from PIL import Image
from spikingjelly.activation_based.functional import reset_net
from torchvision import transforms

PLUGIN_DIR = Path(__file__).resolve().parent
CODE_DIR = PLUGIN_DIR / "deepeisnn_code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models.factory import build_model
from modules.activation import LIF
from modules.norm1d import SpikingEiNorm1d
from modules.norm2d import SpikingEiNorm2d


MODEL_IDENTIFIER = "deepeisnn_cifar10_resnet18_full"
HF_REPO_ID = "vwOvOwv/DeepEISNN"
HF_SUBFOLDER = "CIFAR10-ResNet18"
IMAGE_SIZE = 32
BATCH_SIZE = 64
SEED = 2025


def _patch_xarray_entry_points_compat() -> None:
    try:
        import xarray.backends.plugins as plugins
    except ModuleNotFoundError:
        return

    original_entry_points = plugins.entry_points
    try:
        entry_points_result = original_entry_points()
    except Exception:
        return
    if hasattr(entry_points_result, "get"):
        return

    class EntryPointsCompat:
        def __init__(self, entry_points):
            self._entry_points = entry_points

        def get(self, group, default=None):
            if hasattr(self._entry_points, "select"):
                selected = self._entry_points.select(group=group)
                return list(selected) if selected is not None else default
            return default

    def compatible_entry_points(*args, **kwargs):
        entry_points = original_entry_points(*args, **kwargs)
        if hasattr(entry_points, "get"):
            return entry_points
        return EntryPointsCompat(entry_points)

    plugins.entry_points = compatible_entry_points


def _mark_dynamic_modules_initialized(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "_inited"):
            module._inited = True


def _load_config_and_weights() -> tuple[dict[str, Any], str]:
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="config.json",
        subfolder=HF_SUBFOLDER,
    )
    weight_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="model.safetensors",
        subfolder=HF_SUBFOLDER,
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config, weight_path


def _load_raw_model() -> torch.nn.Module:
    from safetensors.torch import load_model as load_safetensors_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, weight_path = _load_config_and_weights()
    model = build_model(config, device, np.random.default_rng(SEED))
    load_safetensors_model(model, weight_path, strict=True)
    _mark_dynamic_modules_initialized(model)
    model.to(device)
    model.eval()
    return model


def _collect_layer_names(model: torch.nn.Module) -> list[str]:
    layers = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, (SpikingEiNorm2d, SpikingEiNorm1d, LIF)):
            layers.append(name)
            continue
        if module.__class__.__name__ in {"SpikingEiBasicBlock", "SpikingEiBottleneck"}:
            layers.append(name)
            continue
        if name in {"layer1", "layer2", "layer3", "layer4", "maxpool", "adaptive_avgpool"}:
            layers.append(name)
    return layers


def _preprocess_image_paths(image_filepaths: Iterable[str], image_size: int) -> np.ndarray:
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        lambda image: image.unsqueeze(0),
    ])
    images = []
    for image_filepath in image_filepaths:
        with Image.open(image_filepath) as image:
            images.append(preprocess(image.convert("RGB")))
    return np.concatenate(images)


def _output_to_numpy(output: torch.Tensor, temporal_steps: int = 1) -> np.ndarray:
    output = output.detach().float()
    if output.ndim == 5:
        output = output.mean(dim=0)
    elif output.ndim == 3:
        output = output.mean(dim=0)
    elif output.ndim >= 2 and temporal_steps > 1 and output.shape[0] % temporal_steps == 0:
        batch_size = output.shape[0] // temporal_steps
        output = output.view(temporal_steps, batch_size, *output.shape[1:]).mean(dim=0)
    return output.cpu().numpy()


class ResettingSNNPytorchWrapper:
    def __init__(self, model, preprocessing, identifier, batch_size):
        from brainscore_vision.model_helpers.activations.core import (
            ActivationsExtractorHelper,
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._temporal_steps = getattr(model, "T", 1)
        self._extractor = ActivationsExtractorHelper(
            identifier=identifier,
            get_activations=self.get_activations,
            preprocessing=preprocessing,
            batch_size=batch_size,
        )
        self._extractor.insert_attrs(self)

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    def __call__(self, *args, **kwargs):
        return self._extractor(*args, **kwargs)

    def get_layer(self, layer_name):
        module = self._model
        for part in layer_name.split("."):
            module = module._modules.get(part)
            assert module is not None, f"No submodule found for layer {layer_name}"
        return module

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if not torch.is_tensor(output):
                raise TypeError(f"Layer {name} returned non-tensor output.")
            target_dict[name] = _output_to_numpy(output, temporal_steps=self._temporal_steps)

        return layer.register_forward_hook(hook_function)

    def get_activations(self, images, layer_names):
        images = [
            torch.from_numpy(image) if not isinstance(image, torch.Tensor) else image
            for image in images
        ]
        images = torch.stack(images).to(self._device)
        model_dtype = next(self._model.parameters()).dtype
        if images.dtype != model_dtype:
            images = images.to(model_dtype)

        self._model.eval()
        layer_results = OrderedDict()
        hooks = [
            self.register_hook(self.get_layer(layer_name), layer_name, layer_results)
            for layer_name in layer_names
        ]
        try:
            with torch.no_grad():
                reset_net(self._model)
                self._model(images)
                reset_net(self._model)
        finally:
            for hook in hooks:
                hook.remove()
            reset_net(self._model)
        return layer_results

    def layers(self):
        for name, module in self._model.named_modules():
            if len(list(module.children())) > 0:
                continue
            yield name, module


_RAW_MODEL = None
_LAYER_NAMES = None


def _get_raw_model() -> torch.nn.Module:
    global _RAW_MODEL
    if _RAW_MODEL is None:
        _RAW_MODEL = _load_raw_model()
    return _RAW_MODEL


def get_model(name):
    assert name == MODEL_IDENTIFIER
    _patch_xarray_entry_points_compat()
    model = _get_raw_model()
    preprocessing = functools.partial(_preprocess_image_paths, image_size=IMAGE_SIZE)
    activations_model = ResettingSNNPytorchWrapper(
        model=model,
        preprocessing=preprocessing,
        identifier=MODEL_IDENTIFIER,
        batch_size=BATCH_SIZE,
    )
    activations_model.image_size = IMAGE_SIZE
    return activations_model


def get_layers(name):
    assert name == MODEL_IDENTIFIER
    global _LAYER_NAMES
    if _LAYER_NAMES is None:
        _LAYER_NAMES = _collect_layer_names(_get_raw_model())
    return _LAYER_NAMES


def get_behavioral_readout_layer(name):
    assert name == MODEL_IDENTIFIER
    return "final_ei_norm"


def get_bibtex(model_identifier):
    return r"""
@inproceedings{
  liu2026training,
  title={Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition},
  author={Peiyu Liu and Jianhao Ding and Zhaofei Yu},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=U8preGvn5G}
}
"""


if __name__ == "__main__":
    model = _get_raw_model()
    print(MODEL_IDENTIFIER)
    print(_collect_layer_names(model))
