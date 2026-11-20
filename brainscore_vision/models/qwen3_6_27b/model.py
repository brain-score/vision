"""Qwen3.6-27B's pretrained vision encoder for Brain-Score Vision."""

import functools
import json
import logging
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_images
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.utils import fullname


IDENTIFIER = "qwen3.6-27b"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
IMAGE_SIZE = 256
BATCH_SIZE = 4
CHECKPOINT_PREFIX = "model.visual."

# Vision architecture parameters from the pinned Qwen3.6-27B checkpoint.
VISION_CONFIG = {
    "depth": 27,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4304,
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 5120,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}


class QwenVisionEncoder(nn.Module):
    """
    Adapt image tensors to Qwen's flattened-patch vision-tower interface.

    Qwen groups patches by each spatial-merge cell and duplicates still-image
    patches across the temporal-patch axis. The fixed input size gives every
    image the same token count, allowing Brain-Score to retain patch-level
    activations with a presentation-first shape.
    """

    def __init__(self, visual: nn.Module, image_size: int = IMAGE_SIZE):
        super().__init__()
        self.visual = visual
        self.image_size = image_size
        self.batch_size = None

        config = visual.config
        factor = config.patch_size * config.spatial_merge_size
        if image_size % factor:
            raise ValueError(
                f"image size {image_size} must be divisible by patch size "
                f"{config.patch_size} times merge size {config.spatial_merge_size}"
            )

    def _flatten_patches(self, images: torch.Tensor):
        batch_size, channels, height, width = images.shape
        config = self.visual.config
        patch_size = config.patch_size
        merge_size = config.spatial_merge_size
        temporal_patch_size = config.temporal_patch_size

        if channels != config.in_channels:
            raise ValueError(f"expected {config.in_channels} channels, received {channels}")
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"expected {self.image_size}x{self.image_size} images, "
                f"received {height}x{width}"
            )

        grid_h = height // patch_size
        grid_w = width // patch_size
        patches = images.reshape(
            batch_size,
            channels,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        patches = (
            patches.unsqueeze(6)
            .expand(
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                temporal_patch_size,
                -1,
                -1,
            )
            .reshape(
                batch_size * grid_h * grid_w,
                channels * temporal_patch_size * patch_size * patch_size,
            )
        )
        grid_thw = torch.tensor(
            [[1, grid_h, grid_w]],
            dtype=torch.long,
            device=images.device,
        ).repeat(batch_size, 1)
        return patches, grid_thw

    def forward(self, images: torch.Tensor):
        self.batch_size = images.shape[0]
        patches, grid_thw = self._flatten_patches(images)
        return self.visual(hidden_states=patches, grid_thw=grid_thw)


class QwenPytorchWrapper(PytorchWrapper):
    """Brain-Score 2.0 wrapper for Qwen's packed visual-token activations."""

    def __init__(
        self,
        model,
        preprocessing,
        model_loader,
        identifier=None,
        forward_kwargs=None,
        *args,
        **kwargs,
    ):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.getLogger(fullname(self)).debug(f"Using device {self._device}")
        self._model = model
        self._model_loader = model_loader
        self._model_loaded = False
        self._model_load_lock = threading.Lock()

        identifier = identifier or model.__class__.__name__
        self._extractor = self._build_extractor(
            identifier=identifier,
            preprocessing=preprocessing,
            get_activations=self.get_activations,
            *args,
            **kwargs,
        )
        self._extractor.insert_attrs(self)
        self._forward_kwargs = forward_kwargs or {}

    def _ensure_model_loaded(self):
        if self._model_loaded:
            return
        with self._model_load_lock:
            if self._model_loaded:
                return
            self._model_loader()
            meta_parameters = [
                name for name, parameter in self._model.named_parameters() if parameter.is_meta
            ]
            if meta_parameters:
                raise RuntimeError(
                    "Qwen vision weights were not fully loaded: "
                    + ", ".join(meta_parameters[:5])
                )
            self._model = self._model.to(self._device)
            self._model_loaded = True

    def get_activations(self, images, layer_names):
        self._ensure_model_loaded()
        return super().get_activations(images=images, layer_names=layer_names)

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"layer {name} returned unsupported output type {type(output)}")
            if output.ndim != 2:
                raise ValueError(
                    f"layer {name} returned shape {tuple(output.shape)}; "
                    "expected packed [tokens, channels] activations"
                )

            batch_size = self._model.batch_size
            if not batch_size or output.shape[0] % batch_size:
                raise ValueError(
                    f"cannot split {output.shape[0]} tokens across batch size {batch_size}"
                )
            output = output.reshape(batch_size, -1, output.shape[-1])
            target_dict[name] = output.detach().to(device="cpu", dtype=torch.float32).numpy()

        return layer.register_forward_hook(hook_function)


def _build_vision_skeleton():
    from transformers import Qwen3_5VisionModel
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    config = Qwen3_5VisionConfig(**VISION_CONFIG)
    with torch.device("meta"):
        visual = Qwen3_5VisionModel(config)
    return QwenVisionEncoder(visual=visual)


def _load_sharded_vision_weights(
    model,
    index_path,
    download_file,
    safe_open_file,
    checkpoint_prefix: str = CHECKPOINT_PREFIX,
):
    """Load only the visual tensors listed in a sharded checkpoint index."""
    with Path(index_path).open(encoding="utf-8") as index_file:
        checkpoint_index = json.load(index_file)

    checkpoint_to_model_key = {
        checkpoint_key: checkpoint_key.removeprefix(checkpoint_prefix)
        for checkpoint_key in checkpoint_index["weight_map"]
        if checkpoint_key.startswith(checkpoint_prefix)
    }
    expected_keys = set(model.state_dict())
    checkpoint_keys = set(checkpoint_to_model_key.values())
    if expected_keys != checkpoint_keys:
        missing = sorted(expected_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - expected_keys)
        raise RuntimeError(
            "Qwen vision checkpoint does not match the Transformers architecture. "
            f"Missing keys: {missing[:5]}; unexpected keys: {unexpected[:5]}"
        )

    shard_entries = defaultdict(list)
    for checkpoint_key, model_key in checkpoint_to_model_key.items():
        shard = checkpoint_index["weight_map"][checkpoint_key]
        shard_entries[shard].append((checkpoint_key, model_key))

    for shard, entries in sorted(shard_entries.items()):
        shard_path = download_file(filename=shard)
        with safe_open_file(shard_path, framework="pt", device="cpu") as checkpoint:
            shard_state = {
                model_key: checkpoint.get_tensor(checkpoint_key)
                for checkpoint_key, model_key in entries
            }
        model.load_state_dict(shard_state, strict=False, assign=True)

    remaining_meta = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    if remaining_meta:
        raise RuntimeError(
            "Qwen vision checkpoint left parameters on the meta device: "
            + ", ".join(remaining_meta[:5])
        )


def _materialize_vision_buffers(model):
    """Materialize the one non-persistent rotary buffer created on the meta device."""
    rotary = model.rotary_pos_emb
    inv_freq = 1.0 / (
        rotary.theta ** (torch.arange(0, rotary.dim, 2, dtype=torch.float32) / rotary.dim)
    )
    rotary.register_buffer("inv_freq", inv_freq, persistent=False)


def _load_vision_weights(model):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    download_file = functools.partial(
        hf_hub_download,
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
    )
    index_path = download_file(filename="model.safetensors.index.json")
    _load_sharded_vision_weights(
        model=model,
        index_path=index_path,
        download_file=download_file,
        safe_open_file=safe_open,
    )
    _materialize_vision_buffers(model)
    model.eval()


def load_preprocess_images(image_filepaths):
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            lambda image: image.unsqueeze(0),
        ]
    )
    images = [preprocess(image) for image in load_images(image_filepaths)]
    return np.concatenate(images)


def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    model = _build_vision_skeleton()
    wrapper = QwenPytorchWrapper(
        identifier=IDENTIFIER,
        model=model,
        model_loader=functools.partial(_load_vision_weights, model.visual),
        preprocessing=load_preprocess_images,
        batch_size=BATCH_SIZE,
    )
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    return [
        "visual.blocks.3",
        "visual.blocks.7",
        "visual.blocks.12",
        "visual.blocks.17",
        "visual.blocks.22",
        "visual.blocks.26",
    ]


def get_bibtex(model_identifier):
    assert model_identifier == IDENTIFIER
    return """@misc{qwen3.6-27b,
  title = {{Qwen3.6-27B}: Flagship-Level Coding in a {27B} Dense Model},
  author = {{Qwen Team}},
  month = {April},
  year = {2026},
  url = {https://qwen.ai/blog?id=qwen3.6-27b}
}"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
