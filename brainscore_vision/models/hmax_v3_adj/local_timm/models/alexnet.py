import torch
import torch.nn as nn
from typing import Any

from ._builder import build_model_with_cfg
from .registry import register_model


__all__ = ["AlexNet", "alexnet"]

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.in_chans = 3
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class AlexNet_s(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.in_chans = 3
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(4, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(64 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNet1(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        super(AlexNet1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        # 2304 for 160 x 160
        # 4096 for 192 x 192
        # 9216 for 227 x 227
        # 12544 for 270 x 270
        # 16384 for 321 x 321
        # 25600 for 382 x 382
        # 43264 for 454 x 454

        # create a dict
        in_size_dict = {160: 2304,
                        192: 4096,
                        227: 9216,
                        270: 12544,
                        321: 16384,
                        322: 16384,
                        382: 25600,
                        454: 43264}

        print(f"Channel size: {kwargs['channel_size']}")

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_size_dict[kwargs['channel_size']], 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

import torch.utils.model_zoo as model_zoo

@register_model
def alexnet(pretrained=False, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        pass
        # model.load_state_dict(model_zoo.load_url("/oscar/home/npant1/data/npant1/alexnet-owt-7be5be79.pth"))
    return model

@register_model
def alexnet_s(pretrained=False, **kwargs):
    model = AlexNet_s(**kwargs)
    if pretrained:
        pass
        # model.load_state_dict(model_zoo.load_url("/oscar/home/npant1/data/npant1/alexnet-owt-7be5be79.pth"))
    return model


@register_model
def alexnet_nopool(pretrained=False, **kwargs):
    model = AlexNet1(**kwargs)
    if pretrained:
        pass
        # model.load_state_dict(model_zoo.load_url("/oscar/home/npant1/data/npant1/alexnet-owt-7be5be79.pth"))
    return model