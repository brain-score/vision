import torch
from torch import nn
import torchvision

def get_resnet_backend(
    num_classes:int,
    p_channels:int,
    m_channels:int,
    layers:int=18,
    with_voneblock:bool=False,
    tiny:bool=False
    ):
    """Returns ResNet18 or ResNet50 backend.

    :param num_classes (int): number of classes in the classifier
    :param p_channels (int, optional): number of Midget/P cell channels
    :param m_channels (int, optional): number of Parasol/M cell channels
    :param layers (int, optional): number of architecture layers
    :param with_voneblock (bool, optional): whether to remove the first block of the backend
    :param tiny (bool, optional): whether to employ Tiny ImageNet adaptation (64px/2deg input)
    :return: backend model, number of backend in channels
    """
    assert layers in [18, 50]
    backend = torchvision.models.resnet18() if layers == 18 else torchvision.models.resnet50()
    backend.fc = nn.Linear(backend.fc.in_features, num_classes)
    if with_voneblock:
        # When using the VOneBlock, in channels are defined in the bottleneck
        backend_in_channels = backend.layer1[0].conv1.in_channels
        backend = nn.Sequential(
                *list(backend.children())[4:-1],  # Remove first block from ResNet-18
                nn.Flatten(),
                backend.fc
                )
    else:
        backend_in_channels = 3
        in_channels = p_channels + m_channels
        if tiny: backend.conv1.stride = (1, 1)
        conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=backend.conv1.out_channels,
            kernel_size=backend.conv1.kernel_size,
            stride=backend.conv1.stride,
            padding=backend.conv1.padding,
            bias=backend.conv1.bias
            )
        weight = torch.zeros_like(conv1.weight.data)
        weight[:, :min(in_channels, backend_in_channels), :, :] =\
            backend.conv1.weight.data[:, :min(in_channels, backend_in_channels), :, :]
        if weight.size(1) > backend.conv1.weight.data.size(1):
            nn.init.kaiming_normal_(weight[:, in_channels:, :, :], mode="fan_out", nonlinearity="relu")
        conv1.weight.data = weight
        backend.conv1 = conv1
    
    return backend, backend_in_channels


def get_vgg_backend(
    num_classes:int,
    p_channels:int,
    m_channels:int,
    layers:int=16,
    with_voneblock:bool=False,
    tiny:bool=False
    ):
    """Returns VGG16 or VGG19 backend.

    :param num_classes (int): number of classes in the classifier
    :param p_channels (float, optional): number of Midget/P cell channels
    :param m_channels (float, optional): number of Parasol/M cell channels
    :param layers (int, optional): number of architecture layers
    :param with_voneblock (bool, optional): whether to remove the first block of the backend
    :param tiny (bool, optional): whether to employ Tiny ImageNet adaptation (64px/2deg input)
    :return: backend model, number of backend in channels
    """
    assert layers in [16, 19]
    backend = torchvision.models.vgg16() if layers == 16 else torchvision.models.vgg19()
    backend.classifier[-1] = nn.Linear(backend.classifier[-1].in_features, num_classes)
    if tiny:
        backend.features = nn.Sequential(*list(backend.features[:-1]))
        backend.classifier[0] = nn.Linear(in_features=25088, out_features=2048, bias=True)
        backend.classifier[3] = nn.Linear(in_features=2048, out_features=2048, bias=True)
        backend.classifier[6] = nn.Linear(in_features=2048, out_features=200, bias=True)
    if with_voneblock:
        backend_in_channels = backend.features[2].in_channels
        backend.features = nn.Sequential(
                *list(backend.features[2:])
                )
    else:
        backend_in_channels = 3
        in_channels = p_channels + m_channels
        backend.features[0] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
        conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=backend.features[0].out_channels,
            kernel_size=backend.features[0].kernel_size,
            stride=backend.features[0].stride,
            padding=backend.features[0].padding,
            bias=True
            )
        weight = torch.zeros_like(conv1.weight.data)
        weight[:, :min(in_channels, backend_in_channels), :, :] =\
            backend.conv1.weight.data[:, :min(in_channels, backend_in_channels), :, :]
        if weight.size(1) > backend.conv1.weight.data.size(1):
            nn.init.kaiming_normal_(weight[:, in_channels:, :, :], mode="fan_out", nonlinearity="relu")
        conv1.bias.data = backend.features[0].bias.data
        conv1.weight.data = weight
        backend.features[0] = conv1
    
    return backend, backend_in_channels