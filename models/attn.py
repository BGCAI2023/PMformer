import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import math


# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)
#
#     def forward(self, queries, keys, values, attn_mask):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)
#
#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)
#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)
#
#             scores.masked_fill_(attn_mask.mask, -np.inf)
#
#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)
#
#         if self.output_attention:
#             return (V.contiguous(), A)
#         else:
#             return (V.contiguous(), None)


class DynamicSparseAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(DynamicSparseAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.factor = factor

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)


        coarse_scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is not None:
            coarse_scores.masked_fill_(attn_mask.mask, -float('inf'))


        routing_scores = torch.softmax(coarse_scores, dim=-1)
        routing_scores = routing_scores ** self.factor


        dense_scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        dense_scores = torch.softmax(dense_scores, dim=-1)


        combined_scores = (routing_scores + dense_scores) / 2.0


        sparse_scores = combined_scores / combined_scores.sum(dim=-1, keepdim=True)


        V = torch.einsum("bhls,bshd->blhd", sparse_scores, values)

        return V.contiguous(), None

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)


        self.dynamic_attention = DynamicSparseAttention(mask_flag=mask_flag, factor=factor, scale=scale, attention_dropout=attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)


        V, _ = self.dynamic_attention(queries, keys, values, attn_mask)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, patch_size=5, num_patches=20, attention_dropout=0.1, output_attention=False, prune_rate=0.1):
        super(ProbAttention, self).__init__()
        self.mask_flag = mask_flag
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention
        self.prune_rate = prune_rate
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        scores = torch.zeros(B, H, L, L, device=queries.device)


        for _ in range(self.num_patches):
            start = random.randint(0, L - self.patch_size)
            end = start + self.patch_size

            for i in range(start, end):
                for j in range(start, end):
                    scores[:, :, i, j] = (queries[:, i] * keys[:, j]).sum(dim=-1)


        absolute_scores = scores.abs()
        threshold = torch.quantile(absolute_scores, self.prune_rate, dim=-1, keepdim=True)
        scores = torch.where(absolute_scores < threshold, torch.tensor(float('-inf'), device=scores.device), scores)

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))

        
        A = self.dropout(F.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
