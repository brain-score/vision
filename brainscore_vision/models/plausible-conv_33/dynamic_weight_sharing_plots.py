import numpy as np
import argparse
import os


def parse_arguments(args=None):
    """
    Parse the arguments.
    :param args: None or list of str (e.g. ['--device', 'cuda:0']). If None, parses command line arguments. .
    :return: Namespace
    """
    parser = argparse.ArgumentParser(description='Configure the run')

    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--trial', type=int, default=0)

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    return args


class OneLayerLC:
    def __init__(self, n_neurons, kernel_size):
        self.weights = 1.0 + np.random.randn(n_neurons, kernel_size ** 2)  # used to be 1.0 +
        self.weights_init = self.weights.copy()
        self.weight_momentum = np.zeros_like(self.weights)

    def forward(self, x, lr=1e-3, gamma=1e-4, momentum=0.1):
        z = (self.weights * x).sum(axis=-1)  # x of shape n_neurons, kernel_size ** 2
        self.weight_momentum = momentum * self.weight_momentum + \
                               lr * (-((z - z.mean())[:, None] * x) - gamma * (self.weights - self.weights_init))
        self.weights += self.weight_momentum

    def compute_snr(self):
        mean = self.weights.mean(axis=0, keepdims=True)
        var = (self.weights - mean) ** 2
        return (var / mean ** 2).mean()  # var.sum() / (mean ** 2).sum()

    def compute_init_weight_deviation(self):
        return (self.weights.mean(axis=0) * self.weights_init.mean(axis=0)).sum() / \
               np.linalg.norm(self.weights_init.mean(axis=0)) / np.linalg.norm(self.weights.mean(axis=0))
        # return (((self.weights.mean(axis=0) - self.weights_init.mean(axis=0))) ** 2 / self.weights_init.mean(axis=0) ** 2).mean()

class DynamicLayerLC(OneLayerLC):
    def __init__(self, n_neurons, kernel_size, alpha=10):
        super().__init__(n_neurons, kernel_size)
        # self.weights = 1.0 + torch.randn(n_neurons, kernel_size ** 2)
        # self.weights_init = self.weights.clone()
        # self.weight_momentum = torch.zeros_like(self.weights)

        self.r = np.zeros(n_neurons)
        self.r_inh = np.zeros(1)
        self.tau = 30
        self.alpha = alpha

    def forward(self, x, b, lr=1e-3, gamma=1e-4, momentum=0.1):
        z = (self.weights * x).sum(axis=-1)  # x of shape n_neurons, kernel_size ** 2

        delta_r_inh = -self.r_inh - b + self.r.mean()
        self.r += (-self.r + z - self.alpha * self.r_inh + b) / self.tau
        self.r_inh += delta_r_inh / self.tau

        self.weight_momentum = momentum * self.weight_momentum + \
                               lr * (-((self.r - b)[:, None] * x) - gamma * (self.weights - self.weights_init))
        self.weights += self.weight_momentum


def simulate_dynamics(kernel_size, gamma):
    n_neurons = 100
    alpha = 10
    layer = DynamicLayerLC(n_neurons, kernel_size, alpha)

    n_iters = 150 * 2000 * 5

    snr = np.zeros((n_iters + 1))
    # x_recording = np.zeros((n_iters + 1, n_neurons, kernel_size ** 2))
    # r_recording = np.zeros((n_iters + 1, n_neurons + 1))
    weight_deviation = np.zeros((n_iters + 1))

    x = 0
    x_tau = 2  # 5 for 3, 2 for 9

    lr = 0.0003  # 0.0005 for 3, 0.0003 for 9
    momentum = 0.95
    b = 1.0

    snr[0] = layer.compute_snr()
    weight_deviation[0] = layer.compute_init_weight_deviation()

    for i in range(n_iters):
        if i % 150 == 0:  # 200 for 3, 150 for 9
            x = np.tile(np.random.randn(1, kernel_size ** 2), (n_neurons, 1))
        # x = (1 - 1 / x_tau) * x + torch.randn(1, kernel_size ** 2).repeat(n_neurons, 1) / x_tau
        lr_t = lr / np.sqrt(1 + i / x_tau) * (i > 50)
        layer.forward(x, b, lr_t, gamma, momentum)
        # x_recording[i + 1] = x
        # r_recording[i + 1, :-1] = layer.r
        # r_recording[i + 1, -1] = layer.r_inh
        snr[i + 1] = layer.compute_snr()
        weight_deviation[i + 1] = layer.compute_init_weight_deviation()

    return snr, weight_deviation


def collect_data():
    results = np.zeros((3, 3, 150 * 2000 * 5 + 1)) # k, gamma, time
    folder = '/ceph/scratch/romanp/plausible-conv/logs_d_w_sh/'

    for k_i, kernel in enumerate([3, 6, 9]):
        for g_i, gamma in enumerate([0.1, 0.01, 0.001]):
            counter = 0
            for i in range(50):
                name = 'snr_%d_%f_%d.npy' % (kernel, gamma, i)
                if os.path.isfile(folder + 'weight_dev_%d_%f_%d.npy' % (kernel, gamma, i)):
                    counter += 1
                    results[k_i, g_i] += np.load(folder + name, 'r')
                    min_dev = np.load(folder + 'weight_dev_%d_%f_%d.npy' % (kernel, gamma, i), 'r').min()
                    assert min_dev > 0.95, min_dev

                if counter == 10:
                    break
            assert counter == 10
    np.save(folder + 'final_1.npy', results[0] / 10)
    np.save(folder + 'final_2.npy', results[1] / 10)
    np.save(folder + 'final_3.npy', results[2] / 10)


if __name__ == '__main__':
    collect_data()
    # args = parse_arguments()
    # if not os.path.isfile('/ceph/scratch/romanp/plausible-conv/logs_d_w_sh/snr_%d_%f_%d.npy' %
    #                       (args.kernel_size, args.gamma, args.trial)):
    #     snr, weight_dev = simulate_dynamics(args.kernel_size, args.gamma)
    #     np.save('/ceph/scratch/romanp/plausible-conv/logs_d_w_sh/snr_%d_%f_%d.npy' %
    #             (args.kernel_size, args.gamma, args.trial), snr)
    #     np.save('/ceph/scratch/romanp/plausible-conv/logs_d_w_sh/weight_dev_%d_%f_%d.npy' %
    #             (args.kernel_size, args.gamma, args.trial), weight_dev)
