import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class URENetLayer(nn.Module):
    def __init__(self, in_channel, rep_channel, kernel_size, stride, padding_mode, nonlinearity, softplus_beta,
                 skip_connection, allow_bias, share_weight, n_negated, opposite_pair):
        self.in_channel = in_channel
        self.rep_channel = rep_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.nonlinearity = nonlinearity
        self.softplus_beta = softplus_beta
        self.skip_connection = skip_connection
        self.allow_bias = allow_bias
        self.share_weight = share_weight
        self.n_negated = n_negated
        self.opposite_pair = opposite_pair
        super().__init__()
        assert kernel_size % 2 == 1
        assert kernel_size > 2  # Cannot be 1

        if opposite_pair:
            assert (share_weight and n_negated == 0) and self.rep_channel % 2 == 0
            self._weight_encode = nn.Parameter(
                torch.rand([rep_channel // 2, in_channel, kernel_size, kernel_size]) / (kernel_size * np.sqrt(rep_channel)))
            self._bias_encode = nn.Parameter(torch.rand([rep_channel // 2]) / kernel_size) if allow_bias else None
        else:
            self._weight_encode = nn.Parameter(
                torch.rand([rep_channel, in_channel, kernel_size, kernel_size]) / (kernel_size * np.sqrt(rep_channel)))
            self._bias_encode = nn.Parameter(torch.rand([rep_channel]) / kernel_size) if allow_bias else None
            if not share_weight:
                self._weight_decode = nn.Parameter(
                    torch.rand([rep_channel, in_channel, kernel_size, kernel_size]) / (kernel_size * np.sqrt(rep_channel)))

    def corrupt_image(self, image):
        raise NotImplementedError

    def preprocess_image(self, image):
        return image * self.preprocess_gain + self.preprocess_bias

    def weight_encode(self):
        if self.opposite_pair:
            return self._weight_encode.repeat_interleave(2, dim=0) * torch.tensor(
                [1, -1] * (self.rep_channel // 2), device=self._weight_encode.device).reshape(-1, 1, 1, 1)
        else:
            return self._weight_encode

    def bias_encode(self):
        if self.opposite_pair:
            return self._bias_encode.repeat_interleave(2, dim=0)
        else:
            return self._bias_encode

    def weight_decode(self):
        if self.share_weight:
            signs = torch.ones([self.rep_channel, 1, 1, 1], device=self.weight_encode().device)
            signs[self.rep_channel - self.n_negated:] = -1
            return signs * self.weight_encode()
        else:
            return self._weight_decode

    def weight_decode_preprocess(self):
        return self.weight_decode() * self.preprocess_gain

    def encode(self, image):
        image_padded = F.pad(image, [(self.kernel_size - 1) // 2] * 4, mode=self.padding_mode)
        z = F.conv2d(image_padded, self.weight_encode(), bias=self.bias_encode(), stride=self.stride)
        if self.nonlinearity == "softplus":
            z = F.softplus(z, beta=self.softplus_beta)
        elif self.nonlinearity == "sigmoid":
            z = torch.sigmoid(z)
        elif self.nonlinearity == "relu":
            z = F.relu(z)
        else:
            raise ValueError
        return z

    def decode(self, image, odd, return_separate=False):
        groups = self.rep_channel if return_separate else 1
        if self.stride == 1:
            image_padded = F.pad(image, [(self.kernel_size - 1) // 2] * 4, mode=self.padding_mode)
            x_hat = F.conv_transpose2d(image_padded, self.weight_decode(), padding=self.kernel_size - 1, groups=groups)
        elif self.stride == 2:
            pad_start = (self.kernel_size - 1) // 2
            pad_end = pad_start + 1 - np.array(odd)
            pad = np.array([pad_start, pad_end[0], pad_start, pad_end[1]])
            image_prepad = F.pad(image, tuple(pad // 2), mode=self.padding_mode)
            pad_remain = (pad % 2).reshape([2, 2]).sum(axis=1)
            x_hat = F.conv_transpose2d(image_prepad, self.weight_decode(), stride=2,
                                       padding=tuple(self.kernel_size - 1 - pad_remain // 2),
                                       output_padding=tuple(pad_remain % 2), groups=groups)
        if return_separate:
            x_hat = torch.reshape(x_hat,
                                  (x_hat.shape[0], self.rep_channel, self.in_channel, x_hat.shape[2], x_hat.shape[3]))
        return x_hat

    def preprocess_encode(self, image, corrupt=False):
        image = self.preprocess_image(image)
        if corrupt:
            image = self.corrupt_image(image)
        return self.encode(image)

    def loss_func(self, y, x_hat):
        raise NotImplementedError

    def forward(self, y, return_loss=False, return_separate=False):
        assert (not return_loss) or (not return_separate)
        z = self.encode(y)
        odd = torch.tensor(y.shape)[2:4] % 2
        x_hat = self.decode(z, odd, return_separate=return_separate)
        if not return_separate:
            if self.skip_connection:
                x_hat = x_hat + y
            if return_loss:
                loss = self.loss_func(y, x_hat)
                return x_hat, loss
        return x_hat

    def learn_preprocess_parameters(self, train_data):
        raise NotImplementedError

    def train_network(self, train_data, batch_size, n_epoch, learning_rate, display_interval, training_loss):
        self.learn_preprocess_parameters(train_data)
        train_data = self.preprocess_image(train_data)
        if torch.cuda.is_available():
            self.cuda()
            train_data = train_data.cuda()
            print("Training SURE network on GPU...")
        else:
            print("Training SURE network on CPU...")
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        total_loss = 0
        n_batch = len(train_data) // batch_size
        for e in range(n_epoch):
            indices = torch.randperm(len(train_data), device=train_data.device)
            for b in range(n_batch):
                data = train_data[indices[b * batch_size: (b + 1) * batch_size]]
                data_corrupted = self.corrupt_image(data)
                data_hat = self.forward(data_corrupted, return_loss=False)
                ure_loss_mc = self.loss_func(data_corrupted, data_hat)  # TODO
                mse_loss = torch.mean(torch.square(data_hat - data))
                copy_loss = torch.mean(torch.square(data_corrupted - data))

                optimizer.zero_grad()
                if training_loss == "URE":
                    ure_loss_mc.backward()
                elif training_loss == "MSE":
                    mse_loss.backward()
                else:
                    raise ValueError
                optimizer.step()
                with torch.no_grad():
                    total_loss += torch.stack([ure_loss_mc, mse_loss, copy_loss])
                    if (b + 1) % display_interval == 0:
                        mean_loss = total_loss / display_interval
                        print(
                            f"Epoch {e + 1}, batch {b + 1}: "
                            f"ure_loss_mc {mean_loss[0]:.6f}, mse_loss {mean_loss[1]:.6f}, copy_loss {mean_loss[2]:.6f}")
                        total_loss = 0

        if self.share_weight is False or self.n_negated == 0:
            with torch.no_grad():
                train_data = train_data[::10]
                train_data_corrupted = self.corrupt_image(train_data)
                separate_outputs = self.forward(train_data_corrupted, return_separate=True)
                if self.opposite_pair:
                    split_shape = \
                        [separate_outputs.shape[0], self.rep_channel // 2, 2] + list(separate_outputs.shape[2:5])
                    separate_outputs = separate_outputs.reshape(split_shape).sum(dim=2)
                indices = torch.square(separate_outputs).mean(dim=[0, 2, 3, 4]).argsort(descending=True)
                self._weight_encode.data = self._weight_encode.data[indices]
                if self.allow_bias:
                    self._bias_encode.data = self._bias_encode.data[indices]
                if not self.share_weight:
                    self._weight_decode.data = self._weight_decode.data[indices]

        self.cpu()
        print("Training is finished\n")

    def save(self, filename):
        torch.save(self, filename)


class SURENetLayer(URENetLayer):
    def __init__(self, in_channel, rep_channel, kernel_size, stride, padding_mode, nonlinearity, softplus_beta,
                 skip_connection, allow_bias, share_weight, n_negated, opposite_pair, subtract_mean, sigma):
        super().__init__(in_channel, rep_channel, kernel_size, stride, padding_mode, nonlinearity, softplus_beta,
                         skip_connection, allow_bias, share_weight, n_negated, opposite_pair)
        self.subtract_mean = subtract_mean
        self.sigma = sigma

    def corrupt_image(self, image):
        return image + self.sigma * torch.randn_like(image)

    def loss_func(self, y, x_hat):
        if self.sigma < 1e-10:
            return torch.mean(torch.square(x_hat - y))
        eps = 1.4e-4 * self.sigma
        perturb_vector = torch.randn_like(y)
        divergence = torch.mean(perturb_vector * (self.forward(y + eps * perturb_vector) - x_hat)) / eps
        loss = torch.mean(torch.square(x_hat - y)) + (2 * self.sigma**2) * divergence - self.sigma * self.sigma
        return loss

    def learn_preprocess_parameters(self, train_data):
        self.register_buffer("preprocess_gain", 1 / (train_data.amax(
            dim=(2, 3), keepdim=True).mean(dim=0, keepdim=True) + 1e-3))
        # TODO: how to compute gain
        train_data = train_data * self.preprocess_gain
        if self.subtract_mean:
            self.register_buffer("preprocess_bias", -train_data.mean(dim=(0, 2, 3), keepdim=True))
        else:
            self.register_buffer("preprocess_bias", torch.zeros_like(train_data).sum(dim=(0, 2, 3), keepdim=True))


class URENetHierarchy:
    def __init__(self, channel_list, stride_list, model_params):
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.stride_list = stride_list
        self.model_params = model_params
        self.n_layer = len(channel_list) - 1
        assert len(stride_list) == self.n_layer
        self.layers = []
        for i in range(self.n_layer):
            self.layers.append(self.get_layer_class()(in_channel=channel_list[i], rep_channel=channel_list[i+1],
                                                      stride=stride_list[i], **model_params))

    def get_layer_class(self):
        raise NotImplementedError

    def train(self, train_data, train_params):
        for i in range(self.n_layer):
            self.layers[i].train_network(train_data, **train_params)
            if i < self.n_layer - 1:
                with torch.no_grad():
                    train_data = self.layers[i].preprocess_encode(train_data, corrupt=False)  # TODO: corrupt?

    def save(self, filename):
        torch.save(self, filename)


class URENetHierarchyEncode(nn.Module):
    def __init__(self, net: URENetHierarchy, i_layer):
        super().__init__()
        self.net = net
        self.i_layer = i_layer

    def forward(self, img):
        for i in range(self.i_layer + 1):
            img = self.net.layers[i].preprocess_encode(img)
        return img


class SURENetHierarchy(URENetHierarchy):
    def get_layer_class(self):
        return SURENetLayer
