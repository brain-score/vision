import torch
import torch.nn as nn
import torch.nn.functional as F


class CMFeederLayer_Torch(nn.Module):
    def __init__(self, n_modules):
        super(CMFeederLayer_Torch, self).__init__()

        self.n_modules = n_modules

    def forward(self, ip):
        x = ip.repeat(1, self.n_modules, 1, 1)
        return x


class HiPerNetBlock(nn.Module):
    def __init__(self, n_ip, n_op, n_ce, n_op_per_cm, n_modules, stride, use_residual, use_projection = True, use_pooled_proj = True, bn_momentum = 0.05 ):
        super(HiPerNetBlock, self).__init__()

        self.use_residual = use_residual
        self.use_projection = use_projection
        self.use_pooled_proj = use_pooled_proj
        self.n_ce = n_ce

        sqz_n_op = int(n_op / 4)

        self.conv_sqz = nn.Conv2d(n_ip, sqz_n_op, 1, 1, 0, bias=False)
        self.bn_sqz = nn.BatchNorm2d(sqz_n_op, momentum=bn_momentum)
        self.relu_sqz = nn.ReLU(True)

        # self.cm_feeder = CMFeederLayer(n_modules)
        self.cm_feeder = CMFeederLayer_Torch(n_modules)

        self.n_ip_ce = sqz_n_op * n_modules
        self.n_op_ce = n_op_per_cm * n_modules

        # we limit the n_op of last CE layer so that the parameters of the 1x1 expansion layer
        # do not grow overly large if number of modules is very big
        # as a rule of thumb, we set it nearly equal to  n_op / 4
        self.n_op_ce_last = int(round(sqz_n_op / n_modules)) * n_modules

        self.conv_ce = nn.ModuleList()
        self.bn_ce = nn.ModuleList()
        self.relu_ce = nn.ModuleList()

        self.conv_ce.append(nn.Conv2d(self.n_ip_ce, self.n_op_ce, 3, stride, 1, groups=n_modules, bias=False))
        self.bn_ce.append(nn.BatchNorm2d(self.n_op_ce,momentum=bn_momentum))
        self.relu_ce.append(nn.ReLU(True))

        for i in range(n_ce-2):
            self.conv_ce.append(nn.Conv2d(self.n_op_ce, self.n_op_ce, 3, 1, 1, groups=n_modules, bias=False))
            self.bn_ce.append(nn.BatchNorm2d(self.n_op_ce, momentum=bn_momentum))
            self.relu_ce.append(nn.ReLU(True))

        self.conv_ce.append(nn.Conv2d(self.n_op_ce, self.n_op_ce_last, 3, 1, 1, groups=n_modules, bias=False))
        self.bn_ce.append(nn.BatchNorm2d(self.n_op_ce_last, momentum=bn_momentum))
        self.relu_ce.append(nn.ReLU(True))

        self.conv_exp = nn.Conv2d(self.n_op_ce_last, n_op, 1, 1, 0, bias=False)
        self.bn_exp = nn.BatchNorm2d(n_op, momentum=bn_momentum)
        self.relu_exp = nn.ReLU(True)


        if(self.use_projection):
            stride = 1 if(use_pooled_proj) else 2
            self.conv_proj = nn.Conv2d(n_ip, n_op, 1, stride, 0, bias=False)
            self.bn_proj = nn.BatchNorm2d(n_op, momentum=bn_momentum)

    def forward(self, ip):
        x = self.relu_sqz(self.bn_sqz(self.conv_sqz(ip)))
        x = self.cm_feeder(x)

        if(self.use_residual):
            x = self.relu_ce[0](self.bn_ce[0](self.conv_ce[0](x)))

            for i in range(1, self.n_ce-1):
                y = self.bn_ce[i](self.conv_ce[i](x))
                x = self.relu_ce[i](x + y)

            # Last CE needs to handled with care
            # As mentioned previously that we limit the n_op of last CE layer so that
            # the parameters of the 1x1 expansion layer do not grow overly large if number of modules is very big
            # Hence, the n_op for last CE do not match with n_op of the previous CE layer
            # and thus an idenity residual connection is not possible
            # In other words, a residual connection will be used iff n_op of all CE layers is same
            if(self.n_op_ce == self.n_op_ce_last):
                idx = self.n_ce - 1
                y = self.bn_ce[idx](self.conv_ce[idx](x))
                x = self.relu_ce[idx](x + y)
            else:
                idx = self.n_ce - 1
                x = self.relu_ce[idx](self.bn_ce[idx](self.conv_ce[idx](x)))
        else:
            for i in range(self.n_ce):
                x = self.relu_ce[i](self.bn_ce[i](self.conv_ce[i](x)))

        x = self.bn_exp(self.conv_exp(x))

        if(self.use_projection):
            if(self.use_pooled_proj):
                z = F.avg_pool2d(ip, 3, 2, 1)
            z = self.bn_proj(self.conv_proj(z))

            return self.relu_exp(x + z)
        else:
            return self.relu_exp(x + ip)


class HiPerNet(nn.Module):
    def __init__(self):
        super(HiPerNet, self).__init__()

        # 76.76
        self.n_op_res = [256, 512, 1024, 2048]
        self.n_op_per_cm = [32, 64, 128, 256]  # cm --> cortical module
        self.strides = [2, 2, 2, 2]
        self.n_ce = [3, 4, 6, 3]
        self.n_modules   = [4, 4, 4, 4]

        self.use_residual = True
        self.use_projection = True
        self.use_pooled_proj = True

        self.n_blocks = len(self.n_op_res)

        self.n_ip_res = [64]
        for i in range(0,self.n_blocks-1):
            self.n_ip_res.append(self.n_op_res[i])

        self.bn_momentum = 0.05

        self.conv_stem = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn_stem = nn.BatchNorm2d(64, momentum=self.bn_momentum)
        self.relu_stem = nn.ReLU(True)

        self.blocks = nn.ModuleList()

        for i in range(self.n_blocks):
            self.blocks.append(HiPerNetBlock(self.n_ip_res[i],
                                             self.n_op_res[i],
                                             self.n_ce[i],
                                             self.n_op_per_cm[i],
                                             self.n_modules[i],
                                             self.strides[i],
                                             self.use_residual,
                                             self.use_projection,
                                             self.use_pooled_proj,
                                             self.bn_momentum
                                             ))


        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        idx = self.n_blocks - 1
        self.conv_classifier = nn.Conv2d(self.n_op_res[idx], 1000, 1, 1, 0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, ip):
        x = self.relu_stem(self.bn_stem(self.conv_stem(ip)))
        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        x = self.global_pool(x)
        x = self.conv_classifier(x)
        return x

