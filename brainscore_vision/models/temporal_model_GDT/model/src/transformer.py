#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math


class Base(torch.nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    ##config_class = BertConfig
    #load_tf_weights = None
    #base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights. """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)


class TransformerPooling(Base):
    def __init__(
        self, 
        emb_dim, 
        hidden_dim, 
        num_layer=2, 
        dp=0.0, 
        num_head=4, 
        all_state=False, 
        positional_emb=False, 
        num_speed=1, 
        qkv_mha=False
    ):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.transformer = nn.ModuleList([
            Transformer(
                hidden_dim=self.hidden_dim, 
                heads=num_head, 
                dropout=dp, 
                qkv_mha=qkv_mha
            ) for i in range(self.num_layer)
        ])
        self.all_state = all_state

        self.positional_emb = positional_emb
        if self.positional_emb:
            max_step = 64
            # create position embedding for different speeds
            self.pos_embs = nn.ModuleList([nn.Embedding(max_step, hidden_dim) for _ in range(num_speed)])
            #create_sinusoidal_embeddings(max_step, hidden_dim, out=self.position_embeddings)
            # positional ids
            self.register_buffer("pos_ids", torch.arange(max_step).expand((1, -1)))
        self.init_weights()
            

    def forward(self, emb, speed=0):
        # Transfomer Encoder
        device = emb.device
        mask = torch.ones(emb.shape[0], emb.shape[1]).to(device)

        # add positional embeddings
        if self.positional_emb:
            pos_ids = self.pos_ids[:, :emb.shape[1]]
            emb = emb + self.pos_embs[speed](pos_ids.to(device)).expand_as(emb)

        for i in range(self.num_layer):
            emb = self.transformer[i](emb, mask) # B x N x H
        # Pooling (cls)
        if self.all_state:
            return emb
        else:
            return emb[:,0,:] # B x H


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class FFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = F.gelu #if config.gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Transformer(nn.Module):
    def __init__(self, hidden_dim, heads=4, dropout=0.0, qkv_mha=False):
        super().__init__()
        self.mha=MultiHeadAttention(heads, hidden_dim, dp=dropout, qkv_mha=qkv_mha)
        self.norm1=nn.LayerNorm(hidden_dim, eps=1e-12)
        self.ffn=FFN(hidden_dim, 2048, hidden_dim)
        self.norm2=nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = dropout

    def forward(self, x, mask):
        # MHA
        attns = self.mha(x, mask)
        attn = attns[0]
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        x = x + attn
        x = self.norm1(x)
        # FFN
        x = x + self.ffn(x)
        x = self.norm2(x)
        x *= mask.unsqueeze(-1).to(x.dtype)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dp=0.1, qkv_mha=False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dp
        #assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        if qkv_mha:
            self.v_lin = nn.Linear(dim, dim)
            self.out_lin = nn.Linear(dim, dim)
        else:
            self.v_lin = Identity() #nn.Linear(dim, dim)
            self.out_lin = Identity() #nn.Linear(dim, dim)

    def forward(self, input, mask, kv=None, cache=None, head_mask=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen
        else:
            klen = kv.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        #dim_per_head = self.dim
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None: # self attention
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None: # or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)  # (bs, n_heads, qlen, klen)
        
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, qlen, klen)
        #print(scores.shape)
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        outputs = (self.out_lin(context),)
        return outputs
