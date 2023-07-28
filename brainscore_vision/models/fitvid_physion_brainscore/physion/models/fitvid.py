import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as Tr

def kl_divergence(mean1, logvar1, mean2, logvar2, batch_size):
    kld = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                 + torch.square(mean1 - mean2) * torch.exp(-logvar2))
    return kld.sum() / batch_size

class SEBlock(nn.Module):
    """Squeeze-and-Excitation"""
    def __init__(
            self,
            input_size,
            hidden_size,
            act=F.relu,
            dim=(-2,-1),
            dtype=torch.float32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act = act
        self.dim = dim
        self.dtype = dtype

        self.squeeze = nn.Conv2d(self.input_size, self.hidden_size, 1)
        self.expand = nn.Conv2d(self.hidden_size, self.input_size, 1)

    def forward(self, x):
        y = x.mean(self.dim, True)
        y = self.squeeze(y)
        y = self.act(y)
        y = self.expand(y)
        return torch.sigmoid(y) * x

class EncoderBlock(nn.Module):

    def __init__(
            self,
            input_size,
            num_channels,
            kernel_size = 3,
            normalize = nn.BatchNorm2d,
            downsample = False,
            act = torch.nn.SiLU()
    ):
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(self.input_size, self.num_channels, self.kernel_size, stride=self.stride, padding=(self.kernel_size // 2, self.kernel_size // 2), bias=False)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, self.kernel_size, padding='same', bias=False)
        self.shortcut = nn.Conv2d(self.input_size, self.num_channels, 1, stride=self.stride, bias=False)
        self.normalize_input = normalize(self.input_size)
        self.normalize_hidden = normalize(self.num_channels)
        self.normalize_residual = normalize(self.num_channels)
        self.se = SEBlock(self.num_channels, max(self.num_channels // 16, 4))
        self.act = act

    def forward(self, x):

        residual = x
        y = x
        y = self.normalize_input(y)
        y = self.act(y)
        y = self.conv1(y)
        y = self.normalize_hidden(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.se(y)

        if residual.shape != y.shape:
            residual = self.shortcut(residual)
            residual = self.normalize_residual(residual)

        return self.act(y + residual)

class DecoderBlock(nn.Module):

    def __init__(
            self,
            input_size,
            num_channels,
            kernel_size = 5,
            normalize = nn.BatchNorm2d,
            upsample = False,
            expand = 4,
            act = nn.SiLU()
    ):
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.expand = expand
        self.act = act

        self.conv1 = nn.Conv2d(self.input_size,
                               self.num_channels * self.expand,
                               1, bias=False)
        self.conv2 = nn.Conv2d(self.num_channels * self.expand,
                               self.num_channels * self.expand,
                               self.kernel_size, bias=False, padding='same')
        self.conv3 = nn.Conv2d(self.num_channels * self.expand,
                               self.num_channels,
                               1, bias=False)
        self.shortcut = nn.Conv2d(self.input_size,
                                  self.num_channels,
                                  1, bias=False)

        self.normalize_input = normalize(self.input_size)
        self.normalize_hidden1 = normalize(self.num_channels * self.expand)
        self.normalize_hidden2 = normalize(self.num_channels * self.expand)
        self.normalize_output = normalize(self.num_channels)
        self.normalize_residual = normalize(self.num_channels)
        self.se = SEBlock(self.num_channels, max(self.num_channels // 16, 4))


    def upsample_image(self, img, multiplier):
        shape = (img.shape[0],
                 img.shape[1],
                 img.shape[2] * multiplier,
                 img.shape[3] * multiplier)
        R = Tr.Resize(shape[-2:], interpolation=Tr.InterpolationMode.NEAREST)
        return R(img)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_image(x, multiplier=2)

        residual = x
        y = x
        y = self.normalize_input(y)
        y = self.conv1(y)
        y = self.normalize_hidden1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.normalize_hidden2(y)
        y = self.act(y)
        y = self.conv3(y)
        y = self.normalize_output(y)
        y = self.se(y)

        if residual.shape != y.shape:
            residual = self.shortcut(residual)
            residual = self.normalize_residual(y)

        return self.act(y + residual)

class NvaeEncoder(nn.Module):
    def __init__(
            self,
            input_size=3,
            stage_sizes=[2,2,2,2],
            num_channels=64,
            num_classes=256,
            encoder_block=functools.partial(EncoderBlock, downsample=False),
            down_block=functools.partial(EncoderBlock, downsample=True),
            dtype=torch.float32,
            train=True,
            **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.stage_sizes = stage_sizes
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.dtype = dtype
        self.training = train

        self._build_blocks(encoder_block, down_block)

    def _build_blocks(self, encoder_block, down_block):

        blocks = []
        for i, block_size in enumerate(self.stage_sizes):
            block = []
            for j in range(block_size):
                if i == 0 and j == 0:
                    in_ch = self.input_size
                else:
                    in_ch = out_ch + 0
                out_ch = self.num_channels * (2 ** i)
                block_cls = down_block if i > 0 and j == 0 else encoder_block
                block.append(block_cls(in_ch, out_ch))
            block = nn.ModuleList(block)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.linear = nn.Linear(out_ch, self.num_classes)

    def reshape_batch_time(self, x, merge=True):
        shape = list(x.shape)
        if merge:
            try:
                return x.view(self.B * self.T, *shape[2:])
            except:
                return x.reshape(self.B * self.T, *shape[2:])
        else:
            try:
                return x.view(self.B, self.T, *shape[1:])
            except:
                return x.reshape(self.B, self.T, *shape[1:])

    def forward(self, x):

        assert len(x.shape) == 5, x.shape
        self.B, self.T = list(x.shape)[:2]
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.

        x = self.reshape_batch_time(x, merge=True)
        skips = {}
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # print('E', i, j, x.shape)
                block = self.blocks[i][j]
                x = block(x)
                skips[(i,j)] = x.clone()

        # print('E', i, j, x.shape)
        x = x.mean(dim=(-2, -1))
        x = self.linear(x)
        x = self.reshape_batch_time(x, merge=False)
        skips = {k:self.reshape_batch_time(v, False) for k,v in skips.items()}
        # print('lstm inp', x.shape)
        return x, skips


class NvaeDecoder(nn.Module):
    def __init__(self,
                 input_size=256,
                 output_size=3,
                 stage_sizes=[2, 2, 2, 2],
                 num_channels=64,
                 first_block_shape=[512, 8, 8],
                 decoder_block=functools.partial(DecoderBlock, upsample=False),
                 up_block=functools.partial(DecoderBlock, upsample=True),
                 skip_type='residual',
                 train=True,
                 dtype=torch.float32,
                 **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stage_sizes = stage_sizes
        self.num_channels = num_channels
        self.first_block_shape = first_block_shape
        self.skip_type = skip_type
        self.dtype = dtype
        self.training = train

        self.linear = nn.Linear(
            self.input_size,
            np.prod(self.first_block_shape))

        self._build_blocks(decoder_block, up_block)

    def _build_blocks(self, decoder_block, up_block):

        blocks = []
        for i, block_size in enumerate(reversed(self.stage_sizes)):
            block = []
            for j in range(block_size):
                if i == 0 and j == 0:
                    in_ch = self.first_block_shape[0]
                else:
                    in_ch = out_ch + 0
                out_ch = self.num_channels * (2 ** (len(self.stage_sizes)-i-1))
                block_cls = up_block if i > 0 and j == 0 else decoder_block
                block.append(block_cls(in_ch, out_ch))
            block = nn.ModuleList(block)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.output_conv = nn.Conv2d(out_ch, self.output_size, 3,
                                     bias=False, padding='same')


    def reshape_batch_time(self, x, merge=True):
        shape = list(x.shape)
        if merge:
            try:
                return x.view(self.B * self.T, *shape[2:])
            except:
                return x.reshape(self.B * self.T, *shape[2:])
        else:
            try:
                return x.view(self.B, self.T, *shape[1:])
            except:
                return x.reshape(self.B, self.T, *shape[1:])

    def forward(self, x, skips):

        self.B, self.T = x.shape[:2]
        x = self.reshape_batch_time(x, merge=True)
        x = self.linear(x)
        x = x.view(x.shape[0], *self.first_block_shape)

        for i, block_size in enumerate(reversed(self.stage_sizes)):
            for j in range(block_size):
                # print('D', i, j, x.shape)
                block = self.blocks[i][j]
                x = block(x)

                skip_key = (len(self.stage_sizes)-i-1,
                            block_size-j-1)
                if self.skip_type == 'residual':
                    x = x + self.reshape_batch_time(skips[skip_key], merge=True)
                elif self.skip_type == 'concat':
                    x = torch.cat([
                        x,
                        self.reshape_batch_time(skips[skip_key], True)], 1)
                elif self.skip_type is not None:
                    raise ValueError("")

        # print('D', i, j, x.shape)
        x = self.output_conv(x)
        x = torch.sigmoid(x)
        x = self.reshape_batch_time(x, False)
        return x


class MultiGaussianLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size=64,
            output_size=10,
            num_layers=2,
            dtype=torch.float32,
            train=True,
            **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embed = nn.Linear(self.input_size, self.hidden_size)
        self.mean = nn.Linear(self.hidden_size, self.output_size)
        self.logvar = nn.Linear(self.hidden_size, self.output_size)
        self.layers = nn.ModuleList(
            [nn.LSTMCell(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)])

        self.dtype = dtype
        self.training = train

    def init_states(self, batch_size, device):
        states = [None] * self.num_layers
        for i in range(self.num_layers):
            states[i] = (
                torch.zeros((batch_size, self.layers[i].hidden_size)).to(self.dtype).to(device),
                torch.zeros((batch_size, self.layers[i].hidden_size)).to(self.dtype).to(device))
        return states

    def reparametrize(self, mu, logvar):
        var = torch.exp(0.5 * logvar)
        epsilon = torch.randn(var.shape).to(mu.device)
        return mu + var * epsilon

    def forward(self, x, states):
        x = self.embed(x)
        for i in range(self.num_layers):
            states[i] = self.layers[i](x, states[i])
        mean = self.mean(x)
        logvar = self.logvar(x)
        z = self.reparametrize(mean, logvar)
        return states, (z, mean, logvar)

class FitVid(nn.Module):

    def __init__(
            self,
            input_size: int = 3,
            num_channels: int = 64,
            train: bool = True,
            stochastic: bool = True,
            action_conditioned: bool = False,
            z_dim: int = 10,
            g_dim: int = 128,
            rnn_size: int = 256,
            n_past: int = 2,
            beta: float = 1e0,
            dtype: int = torch.float32,
            **kwargs
    ):
        super().__init__()
        self.training = train
        self.stochastic = stochastic
        self.action_conditioned = action_conditioned
        self.input_size = input_size
        self.num_channels = num_channels
        self.z_dim = z_dim
        self.g_dim = g_dim
        self.rnn_size = rnn_size
        self.n_past = n_past
        self.beta = beta
        self.dtype = dtype

        self.encoder = NvaeEncoder(
            input_size=self.input_size,
            num_channels=self.num_channels,
            train=self.training,
            stage_sizes=[2, 2, 2, 2],
            num_classes=self.g_dim)
        self.decoder = NvaeDecoder(
            input_size=self.g_dim,
            output_size=self.input_size,
            num_channels=self.num_channels,
            train=self.training,
            stage_sizes=[2, 2, 2, 2],
            skip_type='residual')
        self.frame_predictor = MultiGaussianLSTM(
            input_size=(self.g_dim + self.z_dim),
            hidden_size=self.rnn_size,
            output_size=self.g_dim,
            num_layers=2)
        self.posterior = MultiGaussianLSTM(
            input_size=self.g_dim,
            hidden_size=self.rnn_size,
            output_size=self.z_dim,
            num_layers=1)
        self.prior = MultiGaussianLSTM(
            input_size=self.g_dim,
            hidden_size=self.rnn_size,
            output_size=self.z_dim,
            num_layers=1)

    def get_input(self, hidden, action, z):
        inp = [hidden]
        if self.action_conditioned:
            inp += [action]
        if self.stochastic:
            inp += [z]
        return torch.cat(inp, dim=1)

    def _broadcast_context_frame_skips(self, skips, frame, num_times):
        """Take the last context frame and broadcast it along the time dimension"""
        def _broadcast(x):
            sh = list(x.shape)
            x = torch.stack([x]*num_times, 1)
            return x
        skips = {
            k: _broadcast(v[:, frame])
            for k,v in skips.items()}
        return skips

    def forward(self, video, actions=None):
        self.B, self.T = video.shape[:2]
        if video.dtype == torch.uint8:
            video = video.to(torch.float32) / 255.0
        pred_s = self.frame_predictor.init_states(self.B, video.device)
        post_s = self.posterior.init_states(self.B, video.device)
        prior_s = self.prior.init_states(self.B, video.device)
        kl = functools.partial(kl_divergence, batch_size=self.B)

        hidden, skips = self.encoder(video)
        skips = self._broadcast_context_frame_skips(
            skips,
            frame=(self.n_past-1),
            num_times=((self.T-1) if self.training else 1))

        kld = torch.tensor(0., dtype=self.dtype).to(video.device)
        means, logvars = [], []
        if self.training:
            h_preds = []
            for t in range(1, self.T):
                h, h_target = hidden[:, t-1], hidden[:, t]
                post_s, (z_t, mu, logvar) = self.posterior(h_target, post_s)
                prior_s, (_, prior_mu, prior_logvar) = self.prior(h, prior_s)
                # print("posterior z %d" % t, z_t.shape)

                if self.action_conditioned:
                    act_t = actions[:, t-1]
                else:
                    act_t = None

                inp = self.get_input(h, act_t, z_t)
                pred_s, (_, h_pred, _) = self.frame_predictor(inp, pred_s)
                # print("frame pred mu %d" % t, h_pred.shape)
                h_pred = torch.sigmoid(h_pred)
                h_preds.append(h_pred)
                means.append(mu)
                logvars.append(logvar)

                kld += kl(mu, logvar, prior_mu, prior_logvar)

            h_preds = torch.stack(h_preds, 1)
            # print("h_preds", h_preds.shape)
            preds = self.decoder(h_preds, skips)

        else: # eval
            preds, x_pred = [], None
            h_preds = []
            for t in range(1, self.T):
                h, h_target = hidden[:, t-1], hidden[:, t]
                if t > self.n_past:
                    h = self.encoder(x_pred.unsqueeze(1))[0][:, 0]

                post_s, (_, mu, logvar) = self.posterior(h_target, post_s)
                prior_s, (z_t, prior_mu, prior_logvar) = self.prior(h, prior_s)

                if self.action_conditioned:
                    act_t = actions[:, t-1]
                else:
                    act_t = None

                inp = self.get_input(h, act_t, z_t)
                pred_s, (_, h_pred, _) = self.frame_predictor(inp, pred_s)
                h_pred = torch.sigmoid(h_pred)
                h_preds.append(h_pred)
                # print("h_pred %d" % t, h_pred.shape)
                x_pred = self.decoder(h_pred.unsqueeze(1), skips)[:,0]

                preds.append(x_pred)
                means.append(mu)
                logvars.append(logvar)
                kld += kl(mu, logvar, prior_mu, prior_logvar)

            h_preds = torch.stack(h_preds, 1)
            preds = torch.stack(preds, 1)

        means = torch.stack(means, 1)
        logvars = torch.stack(logvars, 1)
        mse = nn.MSELoss()(preds, video[:, 1:])
        loss = mse + kld * self.beta

        pad = [0,0] * (preds.dim()-2) + [1,0] + [0,0] # (BS, T, C, H, W) => (W, H, C, T, BS) since torch pad starts from last dim and moves forward
        preds = torch.nn.functional.pad(preds, pad)# add single black frame to beginning of preds to match gt length

        metrics = {
            'hist/mean': means,
            'hist/logvars': logvars,
            'loss/mse': mse,
            'loss/kld': kld,
            'loss/all': loss
        }

        return {
            'loss': loss, 
            'preds': preds, 
            'h_preds': h_preds, 
            'hidden': hidden,
            'metrics': metrics,
            }

