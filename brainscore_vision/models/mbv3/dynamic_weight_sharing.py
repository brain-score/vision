import numpy as np
import argparse


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


class DynamicLayerLC(OneLayerLC):
    def __init__(self, n_neurons, kernel_size, alpha=10):
        super().__init__(n_neurons, kernel_size)

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
        lr_t = lr / np.sqrt(1 + i / x_tau) * (i > 50)
        layer.forward(x, b, lr_t, gamma, momentum)
        # x_recording[i + 1] = x
        # r_recording[i + 1, :-1] = layer.r
        # r_recording[i + 1, -1] = layer.r_inh
        snr[i + 1] = layer.compute_snr()
        weight_deviation[i + 1] = layer.compute_init_weight_deviation()

    return snr, weight_deviation


def simulate_instant():
    # runs all combinations at ones, as it is fast
    gamma_grid = [1e-1, 1e-2, 1e-3]
    kernel_grid = [3, 6, 9]
    n_neurons = 100

    n_iters = 2000
    n_trials = 10

    snr = np.zeros((len(gamma_grid), len(kernel_grid), n_trials, n_iters + 1))
    weight_dev = np.zeros((len(gamma_grid), len(kernel_grid), n_trials, n_iters + 1))

    for gamma_idx, gamma in enumerate(gamma_grid):
        for kernel_idx, kernel_size in enumerate(kernel_grid):
            for trial in range(n_trials):
                layer = OneLayerLC(n_neurons, kernel_size)

                if kernel_size == 18:
                    lr = 0.2
                else:
                    lr = 0.5
                momentum = 0.95

                snr[gamma_idx, kernel_idx, trial, 0] = layer.compute_snr()
                weight_dev[gamma_idx, kernel_idx, trial, 0] = layer.compute_init_weight_deviation()

                for i in range(n_iters):
                    x = np.tile(np.random.randn(1, kernel_size ** 2), (n_neurons, 1))

                    lr_t = lr / (1000 + i)
                    layer.forward(x, lr_t, gamma, momentum)
                    snr[gamma_idx, kernel_idx, trial, i + 1] = layer.compute_snr()
                    weight_dev[gamma_idx, kernel_idx, trial, i + 1] = layer.compute_init_weight_deviation()
