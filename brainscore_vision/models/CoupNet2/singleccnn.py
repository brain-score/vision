import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class CCNN2d(nn.Module):
    def __init__(self, T_n, chans, settled_weight=True, alpha_f=0.1, alpha_l=1.0, alpha_e=1.0, v_e=10.0,
                 kernel_size=3, stride=1, padding=1):
        super(CCNN2d, self).__init__()
        self.alpha_f = alpha_f
        self.alpha_l = alpha_l
        self.alpha_e = alpha_e
        self.v_e = v_e
        self.T = T_n
        self.settled_weight = settled_weight
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kernel = [[0.5, 1.0, 0.5],
                  [1.0, 0.0, 1.0],
                  [0.5, 1.0, 0.5]]
        kernel = torch.Tensor(kernel)
        wei = torch.zeros(chans, chans, 3, 3)
        wei += kernel
        wei = wei
        self.weight = nn.Parameter(data=wei.to(device), requires_grad=False)

        # 卷积
        self.conv = nn.Conv2d(chans, chans, kernel_size, stride, padding)

    def convolution(self, in_put, num):   # x是[N, T, C, W, H]的张量
        # 对f,l,e,u,y初始化
        f = 0
        l = 0
        e = self.v_e/self.alpha_e   # 初始化-1时刻的Ve
        y = torch.zeros_like(in_put[:, 0])   # y的shape是[N,C,W,H]
        y_seq = []   # 用于存储每次迭代的y

        for n in range(num):
            f = np.exp(-self.alpha_f)*f + self.conv(y) + in_put[:, n]
            if self.settled_weight:
                l = np.exp(-self.alpha_l)*l + F.conv2d(y, self.weight, padding=1)
            else:
                l = np.exp(-self.alpha_l) * l + self.conv(y)
            u = f + f * l * 0.5
            e = np.exp(-self.alpha_e)*e+self.v_e*y
            y = torch.sigmoid(u - e)
            y_seq.append(y)
        return torch.stack(y_seq, dim=1)    # 返回shape为[N,T,C,H,W]的y

    def forward(self, x):
        out = self.convolution(x, self.T)
        return out


class CCNN1d(nn.Module):
    def __init__(self, T_n,  features, settled_weight=True, alpha_f=0.1, alpha_l=1.0, alpha_e=1.0, v_e=10.0,
                 kernel_size=3, stride=1, padding=1):
        super(CCNN1d, self).__init__()
        self.alpha_f = alpha_f
        self.alpha_l = alpha_l
        self.alpha_e = alpha_e
        self.v_e = v_e
        self.T = T_n
        self.settled_weight = settled_weight
        self.conv = nn.Conv1d(1, 1, kernel_size, stride, padding, bias=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kernel = [[1.0, 0.0, 1.0]]
        kernel = torch.Tensor(kernel)
        wei = torch.zeros(1, 1, 3)
        wei += kernel
        wei = wei
        self.weight = nn.Parameter(data=wei.to(device), requires_grad=False)

    def convolution(self, in_put, num):  # x的形状[N,T,F]，切片后不会keep dim
        f = 0
        l = 0
        e = self.v_e / self.alpha_e
        y = torch.zeros_like(in_put[:, 0])   # y的shape时[N,F]
        y_seq = []  # 输出给下一层

        for n in range(num):
            f = np.exp(-self.alpha_f) * f + in_put[:, n] + self.conv(y.unsqueeze(dim=1)).squeeze(dim=1)
            if self.settled_weight:
                l = np.exp(-self.alpha_l) * l + F.conv1d(y.unsqueeze(dim=1), self.weight, padding=1).squeeze(dim=1)
            else:
                l = np.exp(-self.alpha_l) * l + self.conv(y.unsqueeze(dim=1)).squeeze(dim=1)
            u = f * (1 + l * 0.5)
            e = np.exp(-self.alpha_e)*e + self.v_e*y
            y = torch.sigmoid(u - e)
            y_seq.append(y)

        return torch.stack(y_seq, dim=1)

    def forward(self, x):
        out = self.convolution(x, self.T)
        return out
