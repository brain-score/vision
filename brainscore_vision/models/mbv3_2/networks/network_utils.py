import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from networks import locally_connected_utils


def get_task_dimensions(task):
    """
    Get the number of channels, channel size (height * width) and number of classes of a dataset.
    :param task: str, '*MNIST' or 'CIFAR*' for any string *
    :return: int, int, int; in_channels, input_size, n_classes
    """
    if 'MNIST' in task:
        in_channels = 1
        input_size = [28, 28]
        n_classes = 10
    elif 'CIFAR' in task:
        in_channels = 3
        input_size = [32, 32]
        n_classes = int(task[5:])
    elif 'ImageNet' == task:
        in_channels = 3
        input_size = [224, 224]
        n_classes = 1000
    elif 'TinyImageNet' == task:
        in_channels = 3
        input_size = [48, 48]
        n_classes = 200
    else:
        raise ValueError('Task must be either *MNIST or CIFAR* or ImageNet or TinyImageNet, but %s was given' % task)
    return in_channels, input_size, n_classes


class Conv2dSizeTracker(nn.Conv2d):
    def __init__(self, position_tracker,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_size = None
        self.position_tracker = position_tracker
        self.position = None

    def extra_repr(self):
        return 'in_size={in_size}, position={position}, '.format(**self.__dict__) + super().extra_repr()

    def forward(self, x):
        self.in_size = x.shape[-2:]  # channel first
        self.position = self.position_tracker()
        return super().forward(x)


class PositionTracker:
    def __init__(self):
        self.current_position = 0

    def __call__(self):
        to_return = self.current_position
        self.current_position += 1
        return to_return


class Conv2dTrackerCreator:
    def __init__(self):
        self.position_tracker = PositionTracker()

    def __call__(self, child):
        layer = Conv2dSizeTracker(self.position_tracker, child.in_channels, child.out_channels,
                                  child.kernel_size, child.stride, child.padding, child.dilation,
                                  child.groups, child.bias is not None, child.padding_mode)
        return layer


def create_locally_connected_layer(child, args):
    assert isinstance(child, Conv2dSizeTracker), 'Need Conv2dSizeTracker for input size data'
    if (child.position < args.n_first_conv) or \
            (args.conv_1x1 and child.kernel_size[0] == 1 and child.kernel_size[1] == 1) or \
            (child.kernel_size[0] == 1 and child.kernel_size[1] == 1
             and child.in_size[0] == 1 and child.in_size[1] == 1 and not args.old_1x1):
        return child
    if args.dynamic_1x1 and child.kernel_size[0] == 1 and child.kernel_size[1] == 1:
        return locally_connected_utils.LocallyConnected1x1HebbWeightSharing(
            child.in_size, child.in_channels, child.out_channels, child.kernel_size,
            args.locally_connected_deviation_eps, child.groups, child.stride, child.bias is not None,
            args.dynamic_sharing_hebb_lr, args.dynamic_sharing_b_freq)
    if args.dynamic_NxN and child.kernel_size[0] != 1:
        return locally_connected_utils.LocallyConnectedNxNHebbWeightSharing(
            child.in_size, child.in_channels, child.out_channels, child.kernel_size,
            args.locally_connected_deviation_eps, child.groups, child.stride, child.bias is not None,
            args.dynamic_sharing_hebb_lr, args.dynamic_sharing_b_freq)
    return locally_connected_utils.LocallyConnected2d(
        child.in_size, child.in_channels, child.out_channels, child.kernel_size,
        args.locally_connected_deviation_eps, child.groups, child.stride, child.bias is not None)


class Conv2dFrozen(nn.Conv2d):
    def __init__(self, weight_val, bias_val, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, padding_mode):
        # nn.Module super().__init__() so weight and bias are not registered as Parameter
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.padding_mode = padding_mode

        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.register_buffer('weight', weight_val)
        self.register_buffer('bias', bias_val)


def freeze_layers(child, args):
    if args.freeze_1x1 and args.conv_1x1 and child.kernel_size[0] == 1 and child.kernel_size[1] == 1:
        with torch.no_grad():
            bias_val = None
            if child.bias is not None:
                bias_val = child.bias.data
            new_child = Conv2dFrozen(child.weight.data, bias_val, child.in_channels, child.out_channels,
                                     child.kernel_size, child.stride, child.padding, child.dilation,
                                     child.groups, child.padding_mode)
        return new_child
    return child


def traverse_module(module, create_conv2d_like_layer):
    attr_to_change = dict()
    for name, child in module.named_children():
        if len(list(child.children())) > 0:
            traverse_module(child, create_conv2d_like_layer)
        else:
            if isinstance(child, torch.nn.Conv2d):
                attr_to_change[name] = create_conv2d_like_layer(child)
    for name, value in attr_to_change.items():
        setattr(module, name, value)


def convert_network_to_locally_connected(net, args):
    in_channels, input_size, n_classes = get_task_dimensions(args.dataset)
    input_size[0], input_size[1] = int(input_size[0] * args.input_scale), int(input_size[1] * args.input_scale)

    traverse_module(net, Conv2dTrackerCreator())
    with torch.no_grad():
        net(torch.ones(1, in_channels, input_size[0], input_size[1]))
    traverse_module(net, lambda x: create_locally_connected_layer(x, args))


def compute_locally_connected_non_padded_weight_average_snr(net):
    snr_list = list()
    for layer in net.modules():
        if isinstance(layer, locally_connected_utils.LocallyConnected2d):
            snr_list.append(layer.compute_non_padded_weight_average_snr())

    return snr_list
