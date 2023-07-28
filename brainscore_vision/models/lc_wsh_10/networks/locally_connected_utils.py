import numpy as np
import torch
import torch.nn as nn


class LocallyConnected2d(nn.Module):
    def __init__(self, in_size, in_channels, out_channels, kernel_size, conv_deviation_eps=0.0,
                 groups=1, stride=1, bias=False):
        super().__init__()

        self.in_size = in_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_deviation_eps = conv_deviation_eps
        if isinstance(kernel_size, int):
            self.kernel_size = kernel_size
        else:
            assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
            self.kernel_size = kernel_size[0]
        if isinstance(stride, int):
            self.stride = stride
        else:
            assert len(stride) == 2 and stride[0] == stride[1]
            self.stride = stride[0]

        self.out_size = [in_size[0] // self.stride, in_size[1] // self.stride]

        # according to kaimin_normal for ReLU with fan_out
        gain = torch.nn.init.calculate_gain('relu')
        # weight_std = 1.0 / np.sqrt(in_channels * kernel_size ** 2)
        weight_std = gain / np.sqrt(out_channels * self.kernel_size ** 2)
        # as in conv2d bias
        bias_std = 1 / np.sqrt(in_channels * self.kernel_size ** 2)

        assert (in_channels % groups == 0) and (out_channels % groups == 0), \
            'In (=%d) and out channels (=%d) must be divisible by groups (=%d)' % (in_channels, out_channels, groups)
        self.groups = groups

        if conv_deviation_eps < 0.0:  # no joint conv
            weights = weight_std * torch.randn(in_channels // groups, out_channels,
                                               self.out_size[0], self.out_size[1],
                                               self.kernel_size, self.kernel_size)
        else:
            channel_init_weight = torch.randn(in_channels // groups, out_channels,
                                              self.kernel_size, self.kernel_size)

            weights = weight_std / np.sqrt(1.0 + conv_deviation_eps ** 2) * (
                channel_init_weight[:, :, None, None, :, :] +
                conv_deviation_eps * torch.randn(in_channels // groups, out_channels,
                                                 self.out_size[0], self.out_size[1],
                                                 self.kernel_size, self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels).uniform_(-bias_std, bias_std))
        else:
            self.bias = None

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.padding_size = (self.kernel_size - 1) // 2
        self.padding = torch.nn.ConstantPad2d(self.padding_size, 0.0)
        # todo: this doesn't work for even kernel sizes

        # neurons that don't receive padded inputs
        self.left_border_h = int(np.ceil((self.padding_size / self.stride)))
        self.right_border_h = (self.in_size[0] - 2 * self.padding_size) // self.stride + self.left_border_h
        self.left_border_w = int(np.ceil((self.padding_size / self.stride)))
        self.right_border_w = (self.in_size[1] - 2 * self.padding_size) // self.stride + self.left_border_w

        self.hebbian_update = 0

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, in_size={in_size}, conv_deviation_eps={conv_deviation_eps} (initial)')
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)

    def get_non_padded_weights(self):
        with torch.no_grad():
            # w: ch_in, ch_out, h_out, w_out, k_h, k_w
            return self.weights.data[:, :, self.left_border_h:self.right_border_h,
                   self.left_border_w:self.right_border_w, :, :].cpu()

    def forward(self, x):
        # x: batch, ch_in, h_out, w_out, k_h, k_w -> z: batch, ch_out, h_out, w_out
        # w: ch_in // groups, ch_out, h_out, w_out, k_h, k_w
        x = self.padding(x)
        x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x = x.view(x.shape[0], self.groups, self.in_channels // self.groups,
                   x.shape[2], x.shape[3], x.shape[4], x.shape[5])

        # g for groups
        output = torch.einsum('bgihwkc,igjhwkc->bgjhw', x,
                            self.weights.view(self.in_channels // self.groups, self.groups,
                                              self.out_channels // self.groups,
                                              self.weights.shape[2], self.weights.shape[3],
                                              self.weights.shape[4], self.weights.shape[5])).contiguous()
        output = output.view(x.shape[0], self.out_channels, self.out_size[0], self.out_size[1])

        if self.bias is not None:
            output += self.bias[None, :, None, None]
        return output

    def compute_non_padded_weight_average_snr(self, tol=1e-20, verbose=True):
        weights_combined = self.get_non_padded_weights()
        # final shape = (in_channels, out_channels, kernel_size, kernel_size)
        weights_mean = torch.mean(weights_combined, dim=(-4, -3))
        weights_std = torch.std(weights_combined, dim=(-4, -3))
        weights_std += (weights_std == 0).float() * tol  # to avoid 0/0

        if verbose:
            print('\tMean abs weight: %f;\tmean std: %f' %  #  (NORMALIZED NEURONS)
                  (weights_mean.abs().mean(), weights_std.mean()))
        return (weights_mean.abs() / weights_std).mean()

    def apply_hebbian_term(self, x, z, hebbian_lr, weight_gamma, saved_weights):
        with torch.no_grad():
            z.add_(-self.compute_lateral_connectivity(z) / \
                                 self.compute_lateral_connectivity(
                                     torch.ones(1, self.out_channels, self.out_size[0], self.out_size[1],
                                                device=z.device)))
            z = torch.einsum('boihwkg,bohw->iohwkg', x, z) / x.shape[0]
            z.add_(self.weights, alpha=weight_gamma)
            z.add_(saved_weights, alpha=-weight_gamma)

            self.weights.data.add_(z, alpha=-hebbian_lr)

    def compute_lateral_connectivity(self, z):
        if self.kernel_size == 1:
            return z.detach().sum(dim=(-2, -1), keepdim=True)

        z_sum = z.detach().clone()

        z_non_padded = z_sum[:, :, self.left_border_h:self.right_border_h, self.left_border_w:self.right_border_w]

        dilation_size = self.kernel_size
        full_size = int(np.ceil(z_non_padded.shape[-1] / dilation_size)) * dilation_size
        z_non_padded_sum = torch.conv2d(z_non_padded,
                                        torch.ones(self.out_channels, 1,
                                                   full_size // dilation_size, full_size // dilation_size,
                                                   device=z.device),
                                        bias=None, dilation=dilation_size,
                                        padding=(full_size - z_non_padded.shape[-1],
                                                 full_size - z_non_padded.shape[-1]),
                                        groups=self.out_channels)[:, :, (full_size - z_non_padded.shape[-1]):,
                           (full_size - z_non_padded.shape[-1]):]
        z_non_padded = torch.kron(
            torch.ones(1, 1, full_size // dilation_size, full_size // dilation_size, device=z.device),
            z_non_padded_sum)[:, :, :z_non_padded.shape[-1], :z_non_padded.shape[-1]]

        z_sum[:, :, self.left_border_h:self.right_border_h, self.left_border_w:self.right_border_w] = z_non_padded

        return z_sum

    def share_weights(self, x, hebbian_lr, weight_gamma, saved_weights):
        with torch.no_grad():
            x = self.padding(x)
            x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
            x = x.view(x.shape[0], self.groups, self.in_channels // self.groups,
                       x.shape[2], x.shape[3], x.shape[4], x.shape[5])

            # g for groups
            output = torch.einsum('bgihwkc,igjhwkc->bgjhw', x,
                                  self.weights.view(self.in_channels // self.groups, self.groups,
                                                    self.out_channels // self.groups,
                                                    self.weights.shape[2], self.weights.shape[3],
                                                    self.weights.shape[4], self.weights.shape[5])).contiguous()
            output = output.view(x.shape[0], self.out_channels, self.out_size[0], self.out_size[1])

            if self.bias is not None:
                output += self.bias[None, :, None, None]

        self.apply_hebbian_term(x, output, hebbian_lr, weight_gamma, saved_weights)

    def share_weights_instantly(self):
        with torch.no_grad():
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    self.weights.data[:, :, i::self.kernel_size, j::self.kernel_size] = \
                        self.weights.data[:, :, i::self.kernel_size, j::self.kernel_size].mean(dim=(2, 3),
                                                                                               keepdim=True)
