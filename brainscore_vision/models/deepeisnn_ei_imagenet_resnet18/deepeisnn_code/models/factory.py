"""Model factory to build different types of models."""

import torch
import numpy as np

from .mlp import SpikingMLP, SpikingEiMLP
from .vgg import SpikingVGG, SpikingEiVGG
from .resnet import SpikingResNet, SpikingEiResNet

in_channels_dict = {
    'MNIST': 1,
    'CIFAR10': 3,
    'CIFAR100': 3,
    'CIFAR10DVS': 2,
    'DVSGesture': 2,
    'TinyImageNet200': 3,
    'ImageNet': 3,
}

n_inputs_dict = {
    'MNIST': 28 * 28,
    'CIFAR10': 32 * 32 * 3,
    'CIFAR100': 32 * 32 * 3,
    'TinyImageNet200': 64 * 64 * 3,
    'ImageNet': 224 * 224 * 3,
}

num_classes_dict = {
    'MNIST': 10,
    'CIFAR10': 10,
    'CIFAR100': 100,
    'CIFAR10DVS': 10,
    'DVSGesture': 11,
    'TinyImageNet200': 200,
    'ImageNet': 1000,
}

def build_model(config: dict, device: torch.device, global_rng: np.random.Generator):
    """Build a model according to the configuration.

    Args:
        config: Configuration dictionary for the model.
        device: Device on which the model will be allocated.
        global_rng: Global random number generator.

    Returns:
        Instantiated model.

    Raises:
        NotImplementedError: If the model type or architecture is unsupported.
        ValueError: If a configuration combination is invalid.
    """
    model_type = config['type']
    T = config['T']
    arch = config['arch']
    num_layers = config['num_layers']
    dataset = config['dataset']
    neuron_config = config['neuron']
    num_classes = num_classes_dict[dataset]
    seq_input = 'DVS' in dataset
    imagenet_backbone = config.get(
        'imagenet_backbone',
        config.get('imagenet_stem', dataset == 'ImageNet'),
    )

    if model_type == 'BN-SNN':
        if arch == 'MLP':
            n_inputs = n_inputs_dict[dataset]
            if seq_input:
                raise ValueError("MLP does not support sequential input currently.")
            return SpikingMLP(
                T,
                num_layers,
                n_inputs,
                num_classes,
                neuron_config,
                BN=True,
            )
        if arch == 'VGG':
            dropout = config['dropout']
            light_classifier = config['light_classifier']
            in_channels = in_channels_dict[dataset]
            return SpikingVGG(
                T,
                num_layers,
                in_channels,
                num_classes,
                neuron_config,
                light_classifier,
                dropout,
                seq_input,
                BN=True,
            )
        if arch == 'ResNet':
            zero_init_residual = config['zero_init_residual']
            in_channels = in_channels_dict[dataset]
            # ResNet has BN by default.
            return SpikingResNet(
                T,
                num_layers,
                in_channels,
                num_classes,
                neuron_config,
                zero_init_residual,
                seq_input,
                imagenet_backbone,
            )
        raise NotImplementedError(
            f"Model {model_type}:{arch}{num_layers} is not implemented."
        )

    if model_type == 'EI-SNN':
        ei_ratio = config['ei_ratio']
        if arch == 'MLP':
            n_inputs = n_inputs_dict[dataset]
            if seq_input:
                raise ValueError("MLP does not support sequential input currently.")
            return SpikingEiMLP(
                T,
                num_layers,
                n_inputs,
                num_classes,
                neuron_config,
                ei_ratio,
                device,
                global_rng,
            )
        if arch == 'VGG':
            dropout = config['dropout']
            light_classifier = config['light_classifier']
            in_channels = in_channels_dict[dataset]
            return SpikingEiVGG(
                T,
                num_layers,
                in_channels,
                num_classes,
                neuron_config,
                light_classifier,
                dropout,
                seq_input,
                ei_ratio,
                device,
                global_rng,
            )
        if arch == 'ResNet':
            zero_init_residual = config['zero_init_residual']
            in_channels = in_channels_dict[dataset]
            return SpikingEiResNet(
                T,
                num_layers,
                in_channels,
                num_classes,
                neuron_config,
                seq_input,
                ei_ratio,
                device,
                global_rng,
                imagenet_backbone,
            )
        raise NotImplementedError(
            f"Model {model_type}:{arch}{num_layers} is not implemented."
        )

    raise NotImplementedError(f"Model type {model_type} is not implemented.")
