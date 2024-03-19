# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
import torch.distributed as dist
from utils.distributed_utils import _gather_from_all


class NCECriterion(nn.Module):
    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem = nLem
        self.register_buffer('avg_exp_score', torch.tensor(-1.))
        self.distributed = dist.is_available() and dist.is_initialized()

    def compute_partition_function(self, out):
        if self.avg_exp_score > 0:
            # Use precomputed value
            return self.avg_exp_score

        with torch.no_grad():
            batch_mean = out.mean().unsqueeze(0)
            if self.distributed:
                batch_mean_gathered = _gather_from_all(batch_mean)
                all_batch_mean = batch_mean_gathered.mean().squeeze()
            else:
                all_batch_mean = batch_mean
            Z = all_batch_mean

        self.avg_exp_score = Z
        return self.avg_exp_score

    def forward(self, scores_pos, scores_neg):
        K = scores_neg.size(1)

        # Compute unnormalized distributions
        exp_scores_pos = torch.exp(scores_pos)
        exp_scores_neg = torch.exp(scores_neg)

        # Compute partition function and normalize
        with torch.no_grad():
            avg_exp_score = self.compute_partition_function(exp_scores_neg)

        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
        Pmt = torch.div(exp_scores_pos, exp_scores_pos + K * avg_exp_score)
        lnPmtSum = -torch.log(Pmt).mean(-1)

        # eq 5.2 : P(origin=noise) = k*Pn / (Pms + k*Pn)
        Pon = torch.div(K * avg_exp_score, exp_scores_neg + K * avg_exp_score)
        lnPonSum = -torch.log(Pon).sum(-1)

        loss = (lnPmtSum + lnPonSum).mean()
        return loss