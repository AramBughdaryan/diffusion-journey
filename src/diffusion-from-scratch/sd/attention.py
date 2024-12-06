import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed, in_proj_bias=True, out_proj_bias=True) -> None:
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.out_proj = nn.Linear(d_embed * n_heads, d_embed)
    
    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: (Batch_size, Seq_len, d_model)
        input_shape = x.shape

        batch_size, seq_length, d_embed = input_shape

        intermim_shape = (batch_size, seq_length, self.n_heads, self.d_head)
        
        # (Batch_size, Seq_len, d_model) -> (Batch_size, Seq_len, d_model) -> 3 tensors of shape (Batch_size, Seq_len, d_model)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_size, Seq_len, d_model) -> (Batch_size, Seq_len, H, d_model / H) -> (Batch_size, H, Seq_len, d_model/H)
        q = q.view(intermim_shape).trasnpose(1, 2)
        k = k.view(intermim_shape).trasnpose(1, 2)
        v = v.view(intermim_shape).trasnpose(1, 2)

        # weight: (Batch_size, H, Seq_len, Seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principle diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            mask.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head) # TODO why is here d_head but not d_model

        # (Batch_size, H, Seq_len, Seq_len) @ (Batch_size, H, Seq_len, d_model / H) -> (Batch_size, H, Seq_len, d_model / H)
        output = weight @ v

        # (Batch_size, Seq_len, H, d_model / H)
        output = output.transpose(1, 2)

        # (Batch_size, Seq_len, d_model)
        output = output.reshape(input_shape)

        # (Batch_size, Seq_len, d_model)
        output = self.out_proj(output)

        return output

        

class CrossAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x: (latent): (Batch_size, Seq_len_Q, Dim_Q)
        # y: (context): (Batch_size, Seq_len_V, Dim_V) = (Batch_size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)

        k = self.k_proj(y)
        v = self.v_proj(y)


        # (Batch_size, n_heads, seq_len, d_head)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_size, n_heads, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch_size, n_heads, seq_len, d_head)
        output = weight @ v

        # (Batch_size, seq_len, n_heads, d_head)
        output = output.transpose(1, 2).contingous()

        #(Batch_size, seq_len, d_embed)
        output = output.view(interim_shape)

        output = self.out_proj(output)

        return output


