# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import pprint
from ..utils.distributed_utils import _gather_from_all
from ..utils.alias_method import AliasMethod
from .nce import NCECriterion

__all__ = ['AVID']


class AVIDSimilarityMemoryBank(nn.Module):
    def __init__(self,
                 memory_size,
                 embedding_dim,
                 xModal=True,
                 wModal=False,
                 num_negatives=1024,
                 momentum=0.5,
                 device=0
                 ):
        super(AVIDSimilarityMemoryBank, self).__init__()
        self.num_negatives = num_negatives
        self.temperature = 0.07
        if not isinstance(momentum, (list, tuple)):
            momentum = [momentum]*2
        self.momentum = momentum
        self.device = device

        self.multinomial = AliasMethod(torch.ones(memory_size-1))
        self.xModal = xModal
        self.wModal = wModal

        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0

        self.init_memory(memory_size, embedding_dim)

    def forward(self, video_emb, audio_emb, y):
        K = int(self.num_negatives)

        # Normalize embeddings
        bs, dim = video_emb.shape
        video_emb = F.normalize(video_emb, p=2, dim=1).view(bs, dim, 1)
        audio_emb = F.normalize(audio_emb, p=2, dim=1).view(bs, dim, 1)

        # Sample memories
        with torch.no_grad():
            video_pos_mem = self.view1_mem[y].view(bs, 1, dim)
            audio_pos_mem = self.view2_mem[y].view(bs, 1, dim)

            idx = self.sample_negatives(y, K).to(video_emb.device)
            video_neg_mem = self.view1_mem[idx].view(bs, K, dim)
            audio_neg_mem = self.view2_mem[idx].view(bs, K, dim)

        # Compute scores
        def compute_scores(context_emb, target_embs, T):
            return [torch.bmm(trg, context_emb).squeeze(-1) / T for trg in target_embs]

        scores = {}
        if self.xModal:
            scores['v2a'] = compute_scores(video_emb, [audio_pos_mem, audio_neg_mem], self.temperature)
            scores['a2v'] = compute_scores(audio_emb, [video_pos_mem, video_neg_mem], self.temperature)

        if self.wModal:
            scores['v2v'] = compute_scores(video_emb, [video_pos_mem, video_neg_mem], self.temperature)
            scores['a2a'] = compute_scores(audio_emb, [audio_pos_mem, audio_neg_mem], self.temperature)

        # Update memory bank
        self.update_memory(video_emb.squeeze(-1), audio_emb.squeeze(-1), y)

        return scores

    def sample_negatives(self, y, K):
        bs = y.shape[0]
        idx = self.multinomial.draw(bs * K).view(bs, -1).to(y.device)
        idx = idx + (idx >= y.unsqueeze(1)).long() # Avoid self
        return idx

    def init_memory(self, num_items, embedding_dim):
        self.register_buffer('view1_mem', torch.randn(num_items, embedding_dim))
        self.register_buffer('view2_mem', torch.randn(num_items, embedding_dim))

        self.view1_mem = F.normalize(self.view1_mem, p=2, dim=1)
        self.view1_mem = self.view1_mem.cuda(self.device)

        self.view2_mem = F.normalize(self.view2_mem, p=2, dim=1)
        self.view2_mem = self.view2_mem.cuda(self.device)

        if self.distributed:
            dist.broadcast(self.view1_mem, 0)
            dist.broadcast(self.view2_mem, 0)
            dist.barrier()

    def update_memory(self, video_emb, audio_emb, y):
        video_mom = float(self.momentum[0])
        audio_mom = float(self.momentum[1])

        # gather embeddings from all gpus
        if self.distributed:
            video_emb_gathered = _gather_from_all(video_emb)
            audio_emb_gathered = _gather_from_all(audio_emb)
            y_gathered = _gather_from_all(y)
        else:
            video_emb_gathered = video_emb
            audio_emb_gathered = audio_emb
            y_gathered = y

        # update audio and video memories
        with torch.no_grad():
            l1_pos = self.view1_mem.index_select(0, y_gathered.view(-1))
            l1_pos.mul_(video_mom)
            l1_pos.add_(torch.mul(video_emb_gathered, 1 - video_mom))
            updated_l1 = F.normalize(l1_pos, p=2, dim=1)
            self.view1_mem.index_copy_(0, y_gathered, updated_l1)

            l2_pos = self.view2_mem.index_select(0, y_gathered.view(-1))
            l2_pos.mul_(audio_mom)
            l2_pos.add_(torch.mul(audio_emb_gathered, 1 - audio_mom))
            updated_l2 = F.normalize(l2_pos, p=2, dim=1)
            self.view2_mem.index_copy_(0, y_gathered, updated_l2)

    def __repr__(self):
        num_negatives = int(self.num_negatives)
        view1_mom = float(self.momentum[0])
        view2_mom = float(self.momentum[1])
        repr_dict = {
            'name': self._get_name(),
            'num_negatives': num_negatives,
            'momentum': [view1_mom, view2_mom],
            'view1_buffer_size': self.view1_mem.shape,
            'view2_buffer_size': self.view2_mem.shape,
        }
        return pprint.pformat(repr_dict, indent=2)


class AVID(nn.Module):
    def __init__(self, num_data, embedding_dim,
                 num_negatives=4096,
                 momentum=0.9,
                 xModal_coeff=1.,
                 wModal_coeff=0.,
                 checkpoint=None,
                 device=0):
        super(AVID, self).__init__()
        '''
        AVID criterion.
        This module receives the output embeddings of the video 
        and audio models, computes their non-linear projections, 
        manages the memory bank and computes the final loss.

        Args:
        - num_data: number of instances in the training set.
        - embedding_dim: output dimension of the non-linear projection.
        - num_negatives: number of negatives to draw from memory bank to compute the NCE loss.
        - momentum: memory bank EMA momemtum parameter.
        - xModal_coeff: coefficient for the cross modal loss. (Cross-AVID: 1.0 | Self-AVID: 0.0 | Joint-AVID: 1.0)
        - wModal_coeff: coefficient for the within modal loss. (Cross-AVID: 0.0 | Self-AVID: 1.0 | Joint-AVID: 1.0)
        - checkpoint: optinally specify a checkpoint path to restore the memory bank and partition function
        '''

        self.nce_average = AVIDSimilarityMemoryBank(
            memory_size=num_data,
            embedding_dim=embedding_dim,
            num_negatives=num_negatives,
            momentum=momentum,
            xModal=xModal_coeff>0.,
            wModal=wModal_coeff>0.,
            device=device
        )
        self.nce_average = self.nce_average.cuda(device)

        sum_coeff = (xModal_coeff + wModal_coeff)
        self.xModal_coeff = xModal_coeff / sum_coeff
        self.wModal_coeff = wModal_coeff / sum_coeff
        self.criterion = NCECriterion(num_data)

        # Restore memory bank and partition function if necessary
        if checkpoint is not None:
            ckp = torch.load(checkpoint, map_location='cpu')['train_criterion']
            state_dict = self.state_dict()

            # Restore memory banks
            state_dict['nce_average.view1_mem'] = ckp['nce_average.view1_mem']
            state_dict['nce_average.view2_mem'] = ckp['nce_average.view2_mem']

            # Restore partition function
            Z = torch.stack([ckp[k] for k in ckp if 'avg_exp_score' in k]).mean()
            for k in state_dict:
                if 'avg_exp_score' in k:
                    state_dict[k] = Z
            self.load_state_dict(state_dict)

    def forward(self, emb1, emb2, target):
        '''
        Args
        - emb1: Video embeddings `(N, D)`
        - emb2: Audio embeddings `(N, D)`
        - taget: Intance labels `(N)`
        '''
        tb_log = {}

        # Compare output embeddings to memory bank embeddings and get scores
        # scores given as: {task: [scores_positives, scores_negatives]}
        scores = self.nce_average(emb1, emb2, target)

        # Compute loss
        xModal_loss, wModal_loss = 0., 0
        for k in scores:
            loss = self.criterion(*scores[k])
            if k in {'v2a', 'a2v'}:
                xModal_loss += loss / 2.
            elif k in {'v2v', 'a2a'}:
                wModal_loss += loss / 2.

            # Tensorboard metrics
            tb_log[f'Loss/{k}'] = loss

        # Tensorboard metrics
        tb_log['Loss/xModal'] = xModal_loss
        tb_log['Loss/wModal'] = wModal_loss

        # Final loss
        total_loss = xModal_loss * self.xModal_coeff + wModal_loss * self.wModal_coeff
        return total_loss, tb_log

    def set_epoch(self, epoch):
        pass
