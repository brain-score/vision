# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import numpy as np
import pprint
from utils.alias_method import AliasMethod
from .nce import NCECriterion
from .avid import AVIDSimilarityMemoryBank

__all__ = ['AVID_CMA']


class CMASampler:
    def __init__(self, video_mem, audio_mem, sampling_args):
        '''
        Class responsible for finding the audio-visual correspondences from the audio and video memory banks.
        Correspondences are computed by calling the sample() method.
        To speed things up, this code will be distributed over different GPUs and synced at the end.

        :param video_mem: Video memory bank
        :param audio_mem: Audio memory bank
        :param sampling_args: Dictionary with two fields.
            `type`: Type of correspondence. Options are `consensus`, `union`, `video` and `audio`.
                    Refer to the paper for their meaning.
            `pos_k`: number of positive correspondences to sample for each instance.
        '''
        self.video_mem = video_mem.cpu()
        self.audio_mem = audio_mem.cpu()
        self.sampling_args = sampling_args

    def sample_instance(self, gpu, q_job, q_data):
        video_mem = self.video_mem.cuda(gpu)
        audio_mem = self.audio_mem.cuda(gpu)

        while True:
            query_idx = q_job.get()
            if query_idx is None:
                break

            # Compute video and audio cosine similarities
            video_sim = torch.mm(video_mem, video_mem[query_idx].t())
            audio_sim = torch.mm(audio_mem, audio_mem[query_idx].t())

            # Compute agreement score
            if self.sampling_args['type'] == 'consensus':
                similarity = torch.stack([video_sim, audio_sim], 0).min(dim=0)[0]
            elif self.sampling_args['type'] == 'union':
                similarity = torch.stack([video_sim, audio_sim], 0).max(dim=0)[0]
            elif self.sampling_args['type'] == 'video':
                similarity = video_sim
            elif self.sampling_args['type'] == 'audio':
                similarity = audio_sim
            else:
                raise ValueError

            # Return top-k correspondences
            pos_sim, pos_idx = torch.topk(similarity, self.sampling_args['pos_k']+1, dim=0, sorted=True)
            pos_index = pos_idx[1:].t().cpu().numpy()   # Avoid self
            pos_index = np.sort(pos_index, axis=1)      # Sort indexes so that negative sampling can be done efficiently

            q_data.put((query_idx, pos_index))
        q_data.put((None, None))

    def sample_dispatcher(self, q_job, workers=80):
        num_items = self.video_mem.shape[0]
        batch_size = 16
        for i in range(0, num_items, batch_size):
            query_idx = list(range(i, min(i+batch_size, num_items)))
            q_job.put(query_idx)

        for _ in range(workers):
            q_job.put(None)

    def sample_gather(self, q_data, workers=80):
        num_items = self.video_mem.shape[0]
        positive_index = np.zeros((num_items, self.sampling_args['pos_k'])).astype(int)
        workers_done, done = 0, 0
        while workers_done < workers:
            query_idx, pos_idx = q_data.get()
            if query_idx is None:
                workers_done += 1
            else:
                done += pos_idx.shape[0]
                positive_index[query_idx] = pos_idx
                if done % (64*1000) == 0:
                    print(f"Done {done}/{num_items}")
        return positive_index

    def sample(self):
        # Compute CMA correspondences. Runs on only one process. Distributes work over all gpus.
        num_workers = torch.cuda.device_count()
        q_job = mp.Queue(maxsize=1000)
        q_data = mp.Queue(maxsize=1000)

        # Start job launcher
        disp = mp.Process(target=self.sample_dispatcher, args=(q_job, num_workers), daemon=True)
        disp.start()

        # Start workers
        workers = []
        for gpu in range(num_workers):
            w = mp.Process(target=self.sample_instance, args=(gpu, q_job, q_data), daemon=True)
            w.start()
            workers += [w]

        # Gather results from workers
        pos_index = self.sample_gather(q_data, num_workers)

        # Wait for all jobs to finish
        [w.join() for w in workers]
        disp.join()
        return pos_index


class AVIDSimilarityPositiveExpansion(AVIDSimilarityMemoryBank):
    def __init__(self,
                 memory_size,
                 embedding_dim,
                 xModalInst=True,
                 wModalInst=False,
                 xModalPos=False,
                 wModalPos=True,
                 num_negatives=1024,
                 num_negatives_within=None,
                 sampling_args=None,
                 momentum=0.5,
                 device=0,
                 ):
        super().__init__(memory_size=memory_size, embedding_dim=embedding_dim, xModal=xModalInst, wModal=wModalInst, num_negatives=num_negatives, momentum=momentum, device=device)
        self.num_negatives_within = num_negatives_within
        self.multinomial = AliasMethod(torch.ones(memory_size - sampling_args['pos_k']))
        self.sampling_args = sampling_args

        self.xModalInst = xModalInst
        self.wModalInst = wModalInst
        self.xModalPos = xModalPos
        self.wModalPos = wModalPos

    def forward(self, video_emb, audio_emb, y):
        bs, dim = video_emb.shape
        video_emb = F.normalize(video_emb, p=2, dim=1).view(bs, dim, 1)
        audio_emb = F.normalize(audio_emb, p=2, dim=1).view(bs, dim, 1)

        # Sample memories
        with torch.no_grad():
            video_self_mem = self.view1_mem[y].view(bs, 1, dim)
            audio_self_mem = self.view2_mem[y].view(bs, 1, dim)

            pos_idx, neg_idx = self.memory_sampling(y)
            video_pos_mem = self.view1_mem[pos_idx]
            audio_pos_mem = self.view2_mem[pos_idx]
            video_neg_mem = self.view1_mem[neg_idx]
            audio_neg_mem = self.view2_mem[neg_idx]

        # Compute scores
        def compute_scores(context_emb, target_embs, T):
            return [torch.bmm(trg, context_emb).squeeze(-1) / T for trg in target_embs]

        # Instance Discrimination
        scores = {}
        if self.xModalInst:  # Cross-modal discrimination
            scores['inst-v2a'] = compute_scores(video_emb, [audio_self_mem, audio_neg_mem], self.temperature)
            scores['inst-a2v'] = compute_scores(audio_emb, [video_self_mem, video_neg_mem], self.temperature)
        if self.wModalInst:  # Within-modal discrimination
            scores['inst-v2a'] = compute_scores(video_emb, [audio_self_mem, audio_neg_mem], self.temperature)
            scores['inst-a2v'] = compute_scores(audio_emb, [video_self_mem, video_neg_mem], self.temperature)

        # Positive Set Discrimination
        if self.xModalPos: # Cross-modal discrimination
            scores['pos-v2a'] = compute_scores(video_emb, [audio_pos_mem, audio_neg_mem], self.temperature)
            scores['pos-a2v'] = compute_scores(audio_emb, [video_pos_mem, video_neg_mem], self.temperature)
        if self.wModalPos: # Within-modal discrimination
            # Potentially reduce number of negatives for within-modal discrimination
            wm_video_neg_mem, wm_audio_neg_mem = video_neg_mem, audio_neg_mem
            if self.num_negatives_within is not None:
                wm_video_neg_mem = video_neg_mem[:, :self.num_negatives_within]
                wm_audio_neg_mem = audio_neg_mem[:, :self.num_negatives_within]
            scores['pos-v2v'] = compute_scores(video_emb, [video_pos_mem, wm_video_neg_mem], self.temperature)
            scores['pos-a2a'] = compute_scores(audio_emb, [audio_pos_mem, wm_audio_neg_mem], self.temperature)

        # Update memory
        self.update_memory(video_emb.squeeze(-1), audio_emb.squeeze(-1), y)
        return scores

    def memory_sampling(self, y):
        # Draw positives
        positive_indexes = self.positive_set[y].long()

        # Draw negatives
        bs = y.shape[0]
        rand_idx = self.multinomial.draw(bs * self.num_negatives).view(bs, -1).to(y.device)

        # Avoid positives while sampling negatives (Positive list is sorted.)
        pos_idx = self.positive_set[y].long()
        ref = pos_idx - torch.range(0, pos_idx.shape[1]-1, dtype=pos_idx.dtype).to(pos_idx.device).unsqueeze(0)
        negative_indexes = rand_idx + (rand_idx.unsqueeze(2) >= ref.unsqueeze(1)).sum(2)

        return positive_indexes, negative_indexes

    def find_correspondences(self):
        if self.sampling_args['pos_k'] <= 0:
            return

        # Find CMA correspondences. Only do this on one process if running in distributed mode and sync at the end.
        positive_set = np.zeros((self.view1_mem.shape[0], self.sampling_args['pos_k'])).astype(int)
        if not self.distributed or self.distributed and self.rank == 0:
            torch.cuda.empty_cache()
            positive_set = CMASampler(self.view1_mem, self.view2_mem, self.sampling_args).sample()

        # Find CMA correspondences. Only do this on one process if running in distributed mode and sync at the end.
        if positive_set is not None:
            self.register_buffer('positive_set', torch.from_numpy(positive_set).int())
            self.positive_set = self.positive_set.cuda(self.device)
            if self.distributed:
                dist.broadcast(self.positive_set, 0)

        if self.distributed:
            dist.barrier()

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


class AVID_CMA(nn.Module):
    def __init__(self, num_data, embedding_dim,
                 num_negatives=1024,
                 num_negatives_within=None,
                 momentum=0.5,
                 xModalInstCoeff=1.,
                 wModalInstCoeff=0.,
                 xModalPosCoeff=0.,
                 wModalPosCoeff=1.,
                 sampling_args=None,
                 checkpoint=None,
                 resample_freq=-1,
                 device=0):
        super(AVID_CMA, self).__init__()
        '''
        AVID_CMA criterion.
        This module receives the output embeddings of the video 
        and audio models, computes their non-linear projections, 
        manages the memory bank, draws positive correspondences, 
        and computes the final loss (weighted average between 
        instance discrimination and positive discrimination losses).

        Args:
        - num_data: number of instances in the training set.
        - embedding_dim: output dimension of the non-linear projection.
        - num_negatives: number of negatives to draw from memory bank to compute the NCE loss.
        - num_negatives_within: optionally reduce the number of negatives for the within-modal loss.
        - momentum: memory bank EMA momentum parameter.
        - xModalInstCoeff: coefficient for the cross modal instance discrimination loss. (AVID-CMA: 1.0)
        - wModalInstCoeff: coefficient for the within modal instance discrimination loss. (AVID-AVID: 0.0)
        - xModalPosCoeff: coefficient for the cross modal positive discrimination loss. (AVID-CMA: 0.0)
        - wModalPosCoeff: coefficient for the within modal positive discrimination loss. (AVID-AVID: 1.0)
        - checkpoint: optionally specify a checkpoint path to restore the memory bank and partition function
        '''

        # first setup the NCEAverage method to get the scores of the output wrt. memory bank negatives
        self.nce_average = AVIDSimilarityPositiveExpansion(
            memory_size=num_data,
            embedding_dim=embedding_dim,
            num_negatives=num_negatives,
            num_negatives_within=num_negatives_within,
            momentum=momentum,
            xModalInst=xModalInstCoeff>0.,
            xModalPos=xModalPosCoeff>0.,
            wModalInst=wModalInstCoeff>0.,
            wModalPos=wModalPosCoeff>0.,
            sampling_args=sampling_args,
            device=device
        )
        self.nce_average = self.nce_average.cuda(device)

        # Loss coefficients
        sum_coeff = xModalInstCoeff + wModalInstCoeff + xModalPosCoeff + wModalPosCoeff
        self.xModalInstCoeff = xModalInstCoeff / sum_coeff
        self.wModalInstCoeff = wModalInstCoeff / sum_coeff
        self.xModalPosCoeff = xModalPosCoeff / sum_coeff
        self.wModalPosCoeff = wModalPosCoeff / sum_coeff

        # Setup loss function
        self.criterion = NCECriterion(num_data)

        # Restore memory bank and partition function from AVID checkpoint
        # Needs to be done before finding correspondences
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

        # Find CMA correspondences
        self.resample_freq = resample_freq
        self.nce_average.find_correspondences()

    def forward(self, emb1, emb2, target):
        '''
        Args
        - emb1: Video embeddings `(N, D)`
        - emb2: Audio embeddings `(N, D)`
        - taget: Intance labels `(N)`
        '''
        tb_log = {}

        # Compare output embeddings to memory bank embeddings and get scores
        scores = self.nce_average(emb1, emb2, target)

        # Compute cross/within modal discrimination losses
        xModalInst_loss, wModalInst_loss, xModalPos_loss, wModalPos_loss = 0., 0., 0., 0.
        for k in scores:
            loss = self.criterion(*scores[k])
            if k in {'inst-v2a', 'inst-a2v'}:
                xModalInst_loss += loss / 2.
            elif k in {'inst-v2v', 'inst-a2a'}:
                wModalInst_loss += loss / 2.
            elif k in {'pos-v2a', 'pos-a2v'}:
                xModalPos_loss += loss / 2.
            elif k in {'pos-v2v', 'pos-a2a'}:
                wModalPos_loss += loss / 2.

            # Metrics for tensorboard
            with torch.no_grad():
                tb_log[f'Loss/{k}'] = loss

        # Compute final loss
        total_loss = xModalInst_loss * self.xModalInstCoeff
        total_loss += wModalInst_loss * self.wModalInstCoeff
        total_loss += xModalPos_loss * self.xModalPosCoeff
        total_loss += wModalPos_loss * self.wModalPosCoeff
        return total_loss, tb_log

    def set_epoch(self, epoch):
        # Recompute CMA correspondences every resample_freq epochs
        if self.resample_freq > 0 and epoch > 0 and epoch % self.resample_freq == 0:
            self.nce_average.find_correspondences()
