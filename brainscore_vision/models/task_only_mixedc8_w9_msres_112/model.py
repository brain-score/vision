"""Self-contained PyTorch export of the matched task-only control model."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper,
    load_preprocess_images,
)


IDENTIFIER = "task_only_mixedc8_w9_msres_112"
WEIGHTS_URL = (
    "https://huggingface.co/meenakshik1993/v4-brainscore-weights/resolve/"
    "ed014b77dfb60e4b0da6315e8381a4325e628f15/"
    "task_only_mixedc8_w9_msres_112-c88f6ec9.pt"
)
LAYERS = [
    "taps.core1",
    "taps.core2",
    "taps.core3",
    "taps.core4",
    "taps.head1",
    "taps.head2",
    "taps.head3",
    "taps.head4",
]


class FixedAntialiasedPool(nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.register_buffer("filter", torch.empty(channels, 1, 5, 5))
        self.stride = (stride, stride)
        self.padding = (2, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            features,
            self.filter,
            stride=self.stride,
            padding=self.padding,
            groups=features.shape[1],
        )


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ExportedC8MultiStageCore(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 144
        self.blocks = nn.ModuleList(
            [conv_block(3, channels), *[conv_block(channels, channels) for _ in range(3)]]
        )
        self.pools = nn.ModuleList(
            [FixedAntialiasedPool(channels, stride) for stride in (2, 2, 2, 1)]
        )
        self.output_channels = channels

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = images
        stages = []
        for index, (block, pool) in enumerate(zip(self.blocks, self.pools)):
            features = pool(block(features))
            if index >= 1:
                stages.append(features)
        return stages


class ConvStage(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class ConvNeXtResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.BatchNorm2d(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.activation = nn.GELU()
        self.project = nn.Conv2d(hidden, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(features)
        residual = self.norm(residual)
        residual = self.expand(residual)
        residual = self.activation(residual)
        residual = self.project(residual)
        return features + self.scale * residual


class BottleneckResidualBlock(nn.Module):
    def __init__(self, channels: int, bottleneck_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, 1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(features + self.scale * self.body(features))


class MultiStageResidualImageNetHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        lateral_channels = 128
        self.lateral = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, lateral_channels, 1, bias=False),
                    nn.BatchNorm2d(lateral_channels),
                    nn.GELU(),
                )
                for _ in range(3)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * lateral_channels, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        self.high_resolution_blocks = nn.Sequential(
            *[ConvNeXtResidualBlock(512) for _ in range(3)]
        )
        self.downsample = ConvStage(512, 768)
        self.low_resolution_blocks = nn.Sequential(
            BottleneckResidualBlock(768, 192),
            BottleneckResidualBlock(768, 192),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.0)
        self.classifier = nn.Linear(768, 1000)

    def forward(self, stage_features: list[torch.Tensor]) -> torch.Tensor:
        target_size = stage_features[-1].shape[-2:]
        projected = []
        for projection, features in zip(self.lateral, stage_features):
            features = projection(features)
            if features.shape[-2:] != target_size:
                features = F.adaptive_avg_pool2d(features, target_size)
            projected.append(features)
        features = self.fuse(torch.cat(projected, dim=1))
        features = self.high_resolution_blocks(features)
        features = self.downsample(features)
        features = self.low_resolution_blocks(features)
        features = self.pool(features).flatten(1)
        return self.classifier(self.dropout(features))


class GlobalTap(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            return F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features


class V4ObjectomeModel(nn.Module):
    """Run the classifier and expose the precommitted pooled stage features."""

    def __init__(self):
        super().__init__()
        self.core = ExportedC8MultiStageCore()
        self.head = MultiStageResidualImageNetHead(self.core.output_channels)
        self.taps = nn.ModuleDict(
            {name: GlobalTap() for name in [*(f"core{i}" for i in range(1, 5)), *(f"head{i}" for i in range(1, 5))]}
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = images
        core_stages = []
        for index, (block, pool) in enumerate(zip(self.core.blocks, self.core.pools), 1):
            features = pool(block(features))
            core_stages.append(features)
            self.taps[f"core{index}"](features)

        target_size = core_stages[-1].shape[-2:]
        projected = []
        for projection, features in zip(self.head.lateral, core_stages[-3:]):
            features = projection(features)
            if features.shape[-2:] != target_size:
                features = F.adaptive_avg_pool2d(features, target_size)
            projected.append(features)
        features = self.head.fuse(torch.cat(projected, dim=1))
        self.taps["head1"](features)
        features = self.head.high_resolution_blocks(features)
        self.taps["head2"](features)
        features = self.head.downsample(features)
        self.taps["head3"](features)
        features = self.head.low_resolution_blocks(features)
        pooled = self.taps["head4"](features)
        return self.head.classifier(self.head.dropout(pooled))


def get_model() -> PytorchWrapper:
    model = V4ObjectomeModel()
    state = torch.hub.load_state_dict_from_url(
        WEIGHTS_URL,
        map_location="cpu",
        check_hash=True,
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    preprocessing = functools.partial(load_preprocess_images, image_size=112)
    wrapper = PytorchWrapper(
        identifier=IDENTIFIER,
        model=model,
        preprocessing=preprocessing,
        batch_size=32,
    )
    wrapper.image_size = 112
    return wrapper
