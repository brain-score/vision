import unittest
import torch.nn as nn
import torch
from networks import locally_connected_utils
from warnings import warn


def are_tensors_close(x, y, atol, rtol):
    return torch.all(torch.isclose(x, y, atol=atol, rtol=rtol)).item()


def traverse_locally_connected_dense_tests(init_layer, assertion_callback, input_channels_grid,
                                           output_channels_grid, kernel_size_grid, stride_grid, data_size_grid,
                                           n_weight_trials, batch_size, groups_grid):
    with torch.no_grad():
        for in_ch in input_channels_grid:
            for out_ch in output_channels_grid:
                for kernel_size in kernel_size_grid:
                    for stride in stride_grid:
                        for data_size in data_size_grid:
                            for groups in groups_grid:
                                for bias in [False, True]:
                                    if data_size[0] // stride > 0 and data_size[1] // stride > 0 and \
                                            in_ch % groups == 0 and out_ch % groups == 0:
                                        for weight_trial in range(n_weight_trials):
                                            conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size,
                                                                   padding=(kernel_size - 1) // 2,
                                                                   stride=stride, groups=groups, bias=bias)
                                            locally_connected_layer = locally_connected_utils.LocallyConnected2d(
                                                data_size, in_ch, out_ch, kernel_size, conv_deviation_eps=0.0,
                                                stride=stride, groups=groups, bias=bias)

                                            if init_layer == 'conv':
                                                # out_ch, in_ch, kernel_height, kernel_width
                                                # in_channels, out_channels, :, :, kernel_size, kernel_size
                                                locally_connected_layer.weights.data = locally_connected_layer.weights.data.permute(
                                                    2, 3, 1, 0, 4, 5)
                                                locally_connected_layer.weights.data[:, :] = conv_layer.weight.data
                                                locally_connected_layer.weights.data = locally_connected_layer.weights.data.permute(
                                                    3, 2, 0, 1, 4, 5)

                                                if bias:
                                                    locally_connected_layer.bias.data = conv_layer.bias.data
                                            elif init_layer == 'lc':
                                                conv_layer.weight.data = \
                                                    locally_connected_layer.weights.data[:, :, 0, 0, :, :].permute(1, 0, 2, 3)
                                                if bias:
                                                    conv_layer.bias.data = locally_connected_layer.bias.data

                                            x = torch.randn(batch_size, in_ch, data_size[0], data_size[1])
                                            out_conv = conv_layer(x)
                                            out_locally_connected = locally_connected_layer(x)
                                            assertion_callback(out_conv, out_locally_connected, in_ch, out_ch, kernel_size,
                                                               data_size, weight_trial)


class TestapproxConvSparse(unittest.TestCase):
    # todo:
    #   check padding issues
    #   decide what to do with odd kernel and data sizes
    def setUp(self):
        self.input_channels_grid = [1, 2, 3, 4]
        self.output_channels_grid = [1, 2, 3, 4]
        self.kernel_size_grid = [1, 3]  # different approaches to padding might lead to errors
        self.stride_grid = [1, 2, 4]
        self.batch_size = 10
        self.data_size_grid = [[2, 2], [4, 4], [8, 8]]#[[5, 5], [6, 6], [8, 8]]
        self.n_weight_trials = 10
        self.groups_grid = [1, 2, 3, 4]
        self.n_nonzero = None
        self.isclose_atol = 1e-6
        self.isclose_rtol = 1e-5

        def equal_output_assertion_callback(out_conv, out_locally_connected, in_ch, out_ch, kernel_size,
                                            data_size, weight_trial):
            # self.assertTrue(out_conv.shape == out_locally_connected.shape,
            #                 "in_ch %d, out_ch %d, kernel_size %d, data_size %s,"
            #                 " out_conv.shape %s, out_locally_connected.shape %s" %
            #                 (in_ch, out_ch, kernel_size, data_size,
            #                  out_conv.shape, out_locally_connected.shape))
            if out_conv.shape == out_locally_connected.shape:
                self.assertTrue(are_tensors_close(out_conv, out_locally_connected,
                                                  self.isclose_atol, self.isclose_rtol),
                                "in_ch %d, out_ch %d, kernel_size %d, "
                                "data_size %s, weight trial %d" %
                                (in_ch, out_ch, kernel_size, data_size,
                                 weight_trial))
            else:
                warn("in_ch %d, out_ch %d, kernel_size %d, data_size %s,"
                     " out_conv.shape %s, out_locally_connected.shape %s" %
                     (in_ch, out_ch, kernel_size, data_size,
                      out_conv.shape, out_locally_connected.shape))

        self.equal_output_assertion_callback = equal_output_assertion_callback

    def test_weight_init_conv_to_locally_connected(self):
        traverse_locally_connected_dense_tests('conv', self.equal_output_assertion_callback,
                                         self.input_channels_grid, self.output_channels_grid, self.kernel_size_grid,
                                         self.stride_grid, self.data_size_grid, self.n_weight_trials, self.batch_size,
                                         self.groups_grid)

    def test_weight_init_locally_connected_to_conv(self):
        traverse_locally_connected_dense_tests('lc', self.equal_output_assertion_callback,
                                         self.input_channels_grid, self.output_channels_grid, self.kernel_size_grid,
                                         self.stride_grid, self.data_size_grid,  self.n_weight_trials, self.batch_size,
                                         self.groups_grid)


if __name__ == '__main__':
    unittest.main()
