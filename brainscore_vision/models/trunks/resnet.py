from pathlib import Path
from typing import List, Dict, Any, Union
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50
from vissl.utils.hydra_config import AttrDict
from vissl.models.trunks import register_model_trunk
from vissl.models.model_helpers import Flatten, Identity

from spacetorch.models.positions import NetworkPositions, LayerPositions
#import sys;sys.path.insert(1,'/home/ozhan/TDANN/spacetorch')
from spacetorch.models.trunks.SOCONV_Class import SOCONV

model_mapping = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,}

LAYER_ORDER = [
    "layer1.0",
    "layer1.1",
    "layer2.0",
    "layer2.1",
    "layer3.0",
    "layer3.1",
    "layer4.0",
    "layer4.1",
]


def get_torchvision_model(arch: str):
    model = model_mapping.get(arch)
    assert model is not None
    return model(pretrained=False, progress=False)


@register_model_trunk("spatial_resnet18")
class SpatialResNet18(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super(SpatialResNet18, self).__init__()

        # get the params trunk takes from the config
        self.model_config = model_config
        trunk_params = self.model_config.TRUNK.TRUNK_PARAMS

        self.positions = self._load_positions(trunk_params.position_dir)

        self.base_model = get_torchvision_model("resnet18")

        # drop the fc so we don't have params that go untrained
        self.base_model.fc = Identity()

        self._feature_blocks = nn.ModuleDict(
            [  # type: ignore
                ("conv1", self.base_model.conv1),
                ("maxpool", self.base_model.maxpool),
                ("layer1_0", self.base_model.layer1[0]),
                ("layer1_1", self.base_model.layer1[1]),
                ("layer2_0", self.base_model.layer2[0]),
                ("layer2_1", self.base_model.layer2[1]),
                ("layer3_0", self.base_model.layer3[0]),
                ("layer3_1", self.base_model.layer3[1]),
                ("layer4_0", self.base_model.layer4[0]),
                ("layer4_1", self.base_model.layer4[1]),
                ("avgpool", self.base_model.avgpool),
            ]
        )

    def _load_positions(self, position_dir: Path) -> Dict[str, LayerPositions]:
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    # VISSL requires this signature for forward passes
    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[Union[torch.Tensor, Dict[str, Any]]]:
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        maxpool = self.base_model.maxpool(x)

        x_1_0 = self.base_model.layer1[0](maxpool)
        x_1_1 = self.base_model.layer1[1](x_1_0)
        x_2_0 = self.base_model.layer2[0](x_1_1)
        x_2_1 = self.base_model.layer2[1](x_2_0)
        x_3_0 = self.base_model.layer3[0](x_2_1)
        x_3_1 = self.base_model.layer3[1](x_3_0)
        x_4_0 = self.base_model.layer4[0](x_3_1)
        x_4_1 = self.base_model.layer4[1](x_4_0)

        x = self.base_model.avgpool(x_4_1)
        flat_outputs = torch.flatten(x, 1)

        spatial_outputs = {
            "layer1_0": (x_1_0, self.positions["layer1.0"]),
            "layer1_1": (x_1_1, self.positions["layer1.1"]),
            "layer2_0": (x_2_0, self.positions["layer2.0"]),
            "layer2_1": (x_2_1, self.positions["layer2.1"]),
            "layer3_0": (x_3_0, self.positions["layer3.0"]),
            "layer3_1": (x_3_1, self.positions["layer3.1"]),
            "layer4_0": (x_4_0, self.positions["layer4.0"]),
            "layer4_1": (x_4_1, self.positions["layer4.1"]),
        }

        return [flat_outputs, spatial_outputs]


@register_model_trunk("spatial_resnet50")
class SpatialResNet50(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super(SpatialResNet50, self).__init__()

        # get the params trunk takes from the config
        self.model_config = model_config
        trunk_params = self.model_config.TRUNK.TRUNK_PARAMS

        self.positions = self._load_positions(trunk_params.position_dir)

        self.base_model = get_torchvision_model("resnet50")
        self._feature_blocks = nn.ModuleDict(
            [  # type: ignore
                ("conv1", self.base_model.conv1),
                ("maxpool", self.base_model.maxpool),
                ("layer1_0", self.base_model.layer1[0]),
                ("layer1_1", self.base_model.layer1[1]),
                ("layer1_2", self.base_model.layer1[2]),
                ("layer2_0", self.base_model.layer2[0]),
                ("layer2_1", self.base_model.layer2[1]),
                ("layer2_2", self.base_model.layer2[2]),
                ("layer2_3", self.base_model.layer2[3]),
                ("layer3_0", self.base_model.layer3[0]),
                ("layer3_1", self.base_model.layer3[1]),
                ("layer3_2", self.base_model.layer3[2]),
                ("layer3_3", self.base_model.layer3[3]),
                ("layer3_4", self.base_model.layer3[4]),
                ("layer3_5", self.base_model.layer3[5]),
                ("layer4_0", self.base_model.layer4[0]),
                ("layer4_1", self.base_model.layer4[1]),
                ("layer4_2", self.base_model.layer4[2]),
                ("avgpool", self.base_model.avgpool),
            ]
        )

    def _load_positions(self, position_dir: Path):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    # VISSL requires this signature for forward passes
    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[Union[torch.Tensor, Dict[str, Any]]]:
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        maxpool = self.base_model.maxpool(x)

        x_1_0 = self.base_model.layer1[0](maxpool)
        x_1_1 = self.base_model.layer1[1](x_1_0)
        x_1_2 = self.base_model.layer1[2](x_1_1)
        x_2_0 = self.base_model.layer2[0](x_1_2)
        x_2_1 = self.base_model.layer2[1](x_2_0)
        x_2_2 = self.base_model.layer2[2](x_2_1)
        x_2_3 = self.base_model.layer2[3](x_2_2)
        x_3_0 = self.base_model.layer3[0](x_2_3)
        x_3_1 = self.base_model.layer3[1](x_3_0)
        x_3_2 = self.base_model.layer3[2](x_3_1)
        x_3_3 = self.base_model.layer3[3](x_3_2)
        x_3_4 = self.base_model.layer3[4](x_3_3)
        x_3_5 = self.base_model.layer3[5](x_3_4)
        x_4_0 = self.base_model.layer4[0](x_3_5)
        x_4_1 = self.base_model.layer4[1](x_4_0)
        x_4_2 = self.base_model.layer4[2](x_4_1)

        x = self.base_model.avgpool(x_4_2)
        flat_outputs = torch.flatten(x, 1)

        spatial_outputs = {
            "layer1_0": (x_1_0, self.positions["layer1.0"]),
            "layer1_1": (x_1_1, self.positions["layer1.1"]),
            "layer1_2": (x_1_2, self.positions["layer1.2"]),
            "layer2_0": (x_2_0, self.positions["layer2.0"]),
            "layer2_1": (x_2_1, self.positions["layer2.1"]),
            "layer2_2": (x_2_2, self.positions["layer2.2"]),
            "layer2_3": (x_2_3, self.positions["layer2.3"]),
            "layer3_0": (x_3_0, self.positions["layer3.0"]),
            "layer3_1": (x_3_1, self.positions["layer3.1"]),
            "layer3_2": (x_3_2, self.positions["layer3.2"]),
            "layer3_3": (x_3_3, self.positions["layer3.3"]),
            "layer3_4": (x_3_4, self.positions["layer3.4"]),
            "layer3_5": (x_3_5, self.positions["layer3.5"]),
            "layer4_0": (x_4_0, self.positions["layer4.0"]),
            "layer4_1": (x_4_1, self.positions["layer4.1"]),
            "layer4_2": (x_4_2, self.positions["layer4.2"]),
        }

        return [flat_outputs, spatial_outputs]


@register_model_trunk("custom_resnet")
class VisslResNet(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super(VisslResNet, self).__init__()

        # get the params trunk takes from the config
        self.model_config = model_config
        self.prune_thresh_by_layer = None
        trunk_config = self.model_config.TRUNK.TRUNK_PARAMS.VisslResNet

        ARCH = trunk_config.ARCH

        self.base_model = get_torchvision_model(ARCH)
        self.base_model.fc = Identity()

        self._feature_blocks = nn.ModuleDict(
            [  # type: ignore
                ("conv1", self.base_model.conv1),
                ("bn1", self.base_model.bn1),
                ("conv1_relu", self.base_model.relu),
                ("maxpool", self.base_model.maxpool),
                ("layer1", self.base_model.layer1),
                ("layer2", self.base_model.layer2),
                ("layer3", self.base_model.layer3),
                ("layer4", self.base_model.layer4),
                ("avgpool", self.base_model.avgpool),
                ("flatten", Flatten(1)),
            ]
        )

    # VISSL requires this signature for forward passes
    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[torch.Tensor]:
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        if self.prune_thresh_by_layer is not None:
            x[x < self.prune_thresh_by_layer[0]] = 0
        x = self.base_model.layer2(x)
        if self.prune_thresh_by_layer is not None:
            x[x < self.prune_thresh_by_layer[1]] = 0
        x = self.base_model.layer3(x)
        if self.prune_thresh_by_layer is not None:
            x[x < self.prune_thresh_by_layer[2]] = 0
        x = self.base_model.layer4(x)
        if self.prune_thresh_by_layer is not None:
            x[x < self.prune_thresh_by_layer[3]] = 0

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)

        return [x]
