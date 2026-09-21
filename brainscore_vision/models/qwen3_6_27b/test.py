import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import brainscore_vision
from brainscore_vision.models.qwen3_6_27b.model import (
    IDENTIFIER,
    QwenPytorchWrapper,
    QwenVisionEncoder,
    _load_sharded_vision_weights,
    _materialize_vision_buffers,
    get_layers,
    load_preprocess_images,
)


class RecordingVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=3,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )
        self.recorded_patches = None
        self.recorded_grid = None

    def forward(self, hidden_states, grid_thw):
        self.recorded_patches = hidden_states
        self.recorded_grid = grid_thw
        return hidden_states


class PackedLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Identity()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.batch_size = 2

    def forward(self, inputs):
        return self.layer(inputs)


class FakeSafeFile:
    def __init__(self, tensors):
        self.tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get_tensor(self, key):
        return self.tensors[key]


def test_layers_span_vision_tower():
    assert get_layers(IDENTIFIER) == [
        "visual.blocks.3",
        "visual.blocks.7",
        "visual.blocks.12",
        "visual.blocks.17",
        "visual.blocks.22",
        "visual.blocks.26",
    ]


def test_flattens_images_in_qwen_patch_order():
    visual = RecordingVisionModel()
    model = QwenVisionEncoder(visual=visual, image_size=4)
    images = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)

    model(images)

    assert visual.recorded_patches.shape == (8, 24)
    assert torch.equal(
        visual.recorded_grid,
        torch.tensor([[1, 2, 2], [1, 2, 2]]),
    )
    patches = visual.recorded_patches.reshape(2, 4, 3, 2, 2, 2)
    assert torch.equal(patches[:, :, :, 0], patches[:, :, :, 1])


def test_patch_layout_matches_transformers_processor(tmp_path):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )

    image_path = tmp_path / "gradient.png"
    pixels = np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3) % 256
    Image.fromarray(pixels.astype(np.uint8)).save(image_path)

    visual = RecordingVisionModel()
    visual.config.patch_size = 16
    visual.config.spatial_merge_size = 2
    model = QwenVisionEncoder(visual=visual, image_size=256)
    model(torch.from_numpy(load_preprocess_images([image_path])))

    processor = Qwen2VLImageProcessor(
        do_resize=False,
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    with Image.open(image_path) as image:
        official = processor(
            images=[image.convert("RGB")],
            return_tensors="pt",
        )

    assert torch.equal(visual.recorded_grid, official["image_grid_thw"])
    torch.testing.assert_close(
        visual.recorded_patches,
        official["pixel_values"],
        rtol=0,
        atol=1e-7,
    )


def test_hook_restores_presentation_dimension():
    model = PackedLayerModel()
    wrapper = QwenPytorchWrapper(
        identifier="test-qwen",
        model=model,
        model_loader=lambda: None,
        preprocessing=None,
        batch_size=2,
    )
    wrapper._model_loaded = True
    target = OrderedDict()
    hook = wrapper.register_hook(model.layer, "layer", target)

    model.layer(torch.arange(24, dtype=torch.bfloat16).reshape(6, 4))
    hook.remove()

    assert target["layer"].shape == (2, 3, 4)
    assert target["layer"].dtype == np.float32


def test_wrapper_extracts_qwen_vision_block():
    from transformers import Qwen3_5VisionModel
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=16,
        num_position_embeddings=64,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    model = QwenVisionEncoder(
        visual=Qwen3_5VisionModel(config),
        image_size=32,
    )
    wrapper = QwenPytorchWrapper(
        identifier="tiny-qwen",
        model=model,
        model_loader=lambda: None,
        preprocessing=None,
        batch_size=2,
    )

    activations = wrapper.get_activations(
        images=[torch.zeros(3, 32, 32), torch.ones(3, 32, 32)],
        layer_names=["visual.blocks.0"],
    )

    assert activations["visual.blocks.0"].shape == (2, 64, 32)
    assert activations["visual.blocks.0"].dtype == np.float32


def test_loads_only_prefixed_visual_weights(tmp_path):
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.weight": "language.safetensors",
                    "model.visual.weight": "vision-1.safetensors",
                    "model.visual.bias": "vision-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    tensors = {
        "vision-1.safetensors": {
            "model.visual.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        },
        "vision-2.safetensors": {
            "model.visual.bias": torch.tensor([5.0, 6.0])
        },
    }
    downloaded = []

    def download_file(filename):
        downloaded.append(filename)
        return tmp_path / filename

    def safe_open_file(path, framework, device):
        assert framework == "pt"
        assert device == "cpu"
        return FakeSafeFile(tensors[Path(path).name])

    with torch.device("meta"):
        model = nn.Linear(2, 2)
    _load_sharded_vision_weights(
        model=model,
        index_path=index_path,
        download_file=download_file,
        safe_open_file=safe_open_file,
    )

    assert downloaded == ["vision-1.safetensors", "vision-2.safetensors"]
    assert torch.equal(model.weight, tensors["vision-1.safetensors"]["model.visual.weight"])
    assert torch.equal(model.bias, tensors["vision-2.safetensors"]["model.visual.bias"])


def test_hydrates_tiny_qwen_vision_model_from_meta(tmp_path):
    from transformers import Qwen3_5VisionModel
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=16,
        num_position_embeddings=64,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    reference = Qwen3_5VisionModel(config).to(dtype=torch.bfloat16)
    checkpoint_tensors = {
        f"model.visual.{key}": tensor.detach().clone()
        for key, tensor in reference.state_dict().items()
    }
    shard_name = "vision.safetensors"
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {key: shard_name for key in checkpoint_tensors}}),
        encoding="utf-8",
    )

    with torch.device("meta"):
        visual = Qwen3_5VisionModel(config)
    assert visual.rotary_pos_emb.inv_freq.is_meta

    _load_sharded_vision_weights(
        model=visual,
        index_path=index_path,
        download_file=lambda filename: tmp_path / filename,
        safe_open_file=lambda path, framework, device: FakeSafeFile(checkpoint_tensors),
    )
    _materialize_vision_buffers(visual)

    assert not visual.rotary_pos_emb.inv_freq.is_meta
    assert all(not parameter.is_meta for parameter in visual.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in visual.parameters())

    model = QwenVisionEncoder(visual=visual, image_size=32)
    output = model(torch.zeros(2, 3, 32, 32, dtype=torch.bfloat16))
    assert output.last_hidden_state.shape == (128, 32)
    assert output.last_hidden_state.dtype == torch.bfloat16


def test_preprocessing_is_fixed_size_and_normalized(tmp_path):
    image_path = tmp_path / "black.png"
    Image.new("RGB", (320, 180), color=(0, 0, 0)).save(image_path)

    result = load_preprocess_images([image_path])

    assert result.shape == (1, 3, 256, 256)
    assert result.dtype == np.float32
    assert np.all(result == -1)


@pytest.mark.travis_slow
@pytest.mark.memory_intense
def test_has_identifier():
    model = brainscore_vision.load_model(IDENTIFIER)
    assert model.identifier == IDENTIFIER
