import torch
import torch.nn as nn
import math


class OpponentChannelInhibition(nn.Module):
    def __init__(self, n_channels):
        super(OpponentChannelInhibition, self).__init__()
        self.n_channels = n_channels
        channel_inds = torch.arange(n_channels, dtype=torch.float32)+1.
        channel_inds = channel_inds-channel_inds[n_channels//2]
        self.channel_inds = torch.abs(channel_inds)
        channel_distances = []
        for i in range(n_channels):
            channel_distances.append(torch.roll(self.channel_inds, i))
        self.channel_distances = nn.Parameter(torch.stack(channel_distances), requires_grad=False)
        self.sigmas = nn.Parameter(torch.rand(n_channels)+(n_channels/8), requires_grad=True)

    def forward(self, x):
        sigmas = torch.clamp(self.sigmas, min=0.5)
        gaussians = (1/(2.5066*sigmas))*torch.exp(-1*(self.channel_distances**2)/(2*(sigmas**2))) # sqrt(2*pi) ~= 2.5066
        gaussians = gaussians/torch.sum(gaussians, dim=0)
        gaussians = gaussians.view(self.n_channels,self.n_channels,1,1)
        weighted_channel_inhibition = nn.functional.conv2d(x, weight=gaussians, stride=1, padding=0)
        return x/(weighted_channel_inhibition+1)
