# https://arxiv.org/pdf/2112.10752.pdf
from msilib import sequence
import os
import gzip
import math
import re
from time import time
import traceback
from functools import lru_cache, reduce
from typing import Callable, List
from collections import namedtuple

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class Normalize(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        self.weight = torch.empty(in_channels)
        self.bias = torch.empty(in_channels)
        self.num_groups = num_groups
    
    def backward(self, x):
        # Reshape for LayerNorm to work as GroupNorm
        # Subtract mean and div stddev
        if self.num_groups == None:
            x = x.layernorm()
        else:
            x = x.reshape(x.shape[0], self.num_groups, -1).layernorm().reshape(x.shape)
        
        # Elementwise_affine on channels
        if len(x.shape) == 4:
            return (x * self.weight.reshape(1, -1, 1, 1)) + self.bias.reshape(1, -1, 1, 1)
        else:
            return x.linear(self.weight, self.bias)

class AttentionBlock:
    def __init__(self, in_channels):
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def __call__(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # Compute Attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1) # b, hw, c
        k = k.reshape(b, c, h*w) # b, c, hw
        w_ = q @ k
        w_ = w_ * (c**(-0.5))
        w_ = w_.softmax()

        # Attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)
        h_ = v @ w_
        h_ = h_.reshape(b, c, h, w)

        return x + self.proj_out(h_)
"""
# ************** AutoEncoder KL ****************
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        self.norm = nn.GroupNorm(num_channels=in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # Compute Attention
        b, c, h, w = q.shape
        q = torch.reshape(q, (b, c, h*w))
        # q = q.reshape(b, c, h*w)
        q = torch.permute(q, (0, 2, 1))  # b, hw, c
        # q = q.permute(0, 2, 1)  
        k = torch.reshape(k, (b, c, h*w))  # b, c, hw
        # k = k.reshape(b, c, h*w)  
        w_ = q @ k
        w_ = w_ * (c**(-0.5))
        w_ = torch.softmax(w_)
        # w_ = w_.softmax()

        # Attend to values
        v = torch.reshape(v, (b, c, h*w))
        v = v.reshape(b, c, h*w)
        # w_ = w_.permute(0, 2, 1)
        w_ = torch.permute(w_, (0, 2, 1))
        h_ = v @ w_
        h_ = torch.reshape(h_, (b, c, h, w))
        # h_ = h_.reshape(b, c, h, w)

        return x + self.proj_out(h_)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        self.norm1 = nn.GroupNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(x)))

        return self.nin_shortcut(x) + h


class Mid(nn.Module):
    def __init__(self, block_in):
        self.block1 = ResnetBlock(block_in, block_in)
        self.attn1 = AttentionBlock(block_in)
        self.block2 = ResnetBlock(block_in, block_in)

    def forward(self, x):
        return nn.Sequential([
            self.block1,
            self.attn1,
            self.block2
        ])(x)


class Decoder(nn.Module):
    def __init__(self):
        size = [(128, 256), (256, 512), (512, 512), (512, 512)]
        self.conv_in = nn.Conv2d(4, 512, 3, padding=1)
        self.mid = Mid(512)

        layers = []
        for i, s in enumerate(size):
            layers.append({"block": 
                [ResnetBlock(s[1], s[0]),
                 ResnetBlock(s[0], s[0]),
                 ResnetBlock(s[0], s[0])]
            })
            if i != 0:
                layers[-1]['upsample'] = {'conv': nn.Conv2d(s[0], s[0], 3, padding=1)}
        self.up = layers

        self.norm_out = nn.GroupNorm(128)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid(x)

        for l in self.up[::-1]:
            print('decode', x.shape)
            for b in l['block']: 
                x = b(x)
            if 'upsample' in l:
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html 
                bs, c, py, px = x.shape
                x = torch.reshape(x, (bs, c, py, 1, px, 1)).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
                x = l['upsample']['conv'](x)
        
        return self.conv_out(F.silu(self.norm_out(x)))

class Encoder(nn.Module):
    def __init__(self):
        sz = [(128, 128), (128, 256), (256, 512), (512, 512)]
        self.conv_in = nn.Conv2d(3, 128, 3, padding=1)

        layers = []
        for i, s in enumerate(sz):
            layers.append({"block": 
                [ResnetBlock(s[0], s[1]),
                 ResnetBlock(s[1], s[1]),
                ]
            })
            if i != 3:
                layers[-1]['downsample'] = {'conv': nn.Conv2d(s[1], s[1], 3, stride=2, padding=(0, 1, 0, 1))}
        self.down = layers    
        self.mid = Mid(512)
        self.norm_out = nn.GroupNorm(512)
        self.conv_out = nn.Conv2d(512, 8, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)

        for l in self.down:
            print("decode", x.shape)
            for b in l['block']:
                x = b(x)
            if 'downsample' in l:
                x = l['downsample']['conv'](x)
        x = self.mid(x)

        return self.conv_out(nn.SiLU(self.norm_out(x)))

class AutoEncoder(nn.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(8, 8, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)
    
    def forward(self, x):
        latent = self.encoder(x)
        latent = self.quant_conv(latent)
        latent = latent[:, 0:4]
        print('latent', latent.shape)
        latent = self.post_quant_conv(latent)

        return self.decoder(latent)

# ************** Diffusion Model ******************
class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, out_channels):
        self.in_layers = [
            nn.GroupNorm(channels),
            F.silu,
            nn.Conv2d(channels, out_channels, 3, padding=1)
        ]
        self.emb_layers = [
            F.silu,
            nn.Conv2d(emb_channels, out_channels)
        ]
        self.out_layers = [
            nn.GroupNorm(out_channels),
            F.silu,
            lambda x: x,
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        ]
        self.skip_connections = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else lambda x: x

    def forward(self, x, emb):
        h = nn.Sequential(self.in_layers)(x)
        emb_out = nn.Sequential(self.emb_layers)(emb)
        h = h + torch.reshape(emb_out, (*emb_out.shape, 1, 1))
        h = nn.Sequential(self.out_layers)(h)
        ret = self.skip_connections(x) + h

        return ret

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        self.to_q = nn.Linear(query_dim, n_heads*d_head, bias=False)
        self.to_k = nn.Linear(context_dim, n_heads*d_head, bias=False)
        self.to_v = nn.Linear(query_dim, n_heads*d_head, bias=False)
        self.scale = d_head ** -0.5
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = [nn.Linear(n_heads*d_head, query_dim)]

    """
    def sequential(self, ll: List[Callable[[torch.Tensor], torch.Tensor]]):
        return reduce(lambda x, f: f(x), ll, self)
    """
    # Try this !!!
    def sequential(self, x, layers):
        for l in layers:
            x = l(x)
        return x

    def forward(self, x, context=None):
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q = torch.reshape(q, shape=(x.shape[0], -1, self.num_heads, self.head_size)).permute(0, 2, 1, 3)
        k = torch.reshape(k, shape=(x.shape[0], -1, self.num_heads, self.head_size)).permute(0, 2, 3, 1)
        v = torch.reshape(v, shape=(x.shape[0], -1, self.num_heads, self.head_size)).permute(0, 2, 1, 3)

        score = q.dot(k) * self.scale
        weights = F.softmax(score)
        attention = weights.dot(v).permute(0, 2, 1, 3)
        h_ = attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size))
        
        return self.sequential(h_, self.to_out)
class GeGELU(nn.Module):
    def __init__(self, dim_in, dim_out):
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.dim_out = dim_out

    def forward(self, x):
        x, gate = torch.chunk(self.proj(x), chunks=2, dim=-1)
        
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        self.net = [
            GeGELU(dim, dim*mult),
            lambda x: x,
            nn.Linear(dim*mult, dim)
        ]

    # Try this !!!
    def sequential(self, x, layers):
        for l in layers:
            x = l(x)
        return x

    def forward(self, x):
        self.sequential(x, self.net) 


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, context_dim, n_heads, d_head):
        self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(dim, context_dim, n_heads, d_head)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x

        return x

class SpatialTransformer(nn.Module):
    def __init__(self, channels, context_dim, n_heads, d_head):
        self.norm = nn.GroupNorm(channels)
        assert channels == n_heads * d_head
        self.proj_in = nn.Conv2d(channels, n_heads * d_head, 1)
        self.transformer_blocks = [
            BasicTransformerBlock(channels, context_dim, n_heads, d_head)
        ]
        self.proj_out = nn.Conv2d(n_heads * d_head, channels, 1)
    
    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = torch.reshape(x, (b, c, h*w)).permute(b, c, h, w)
        
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        ret = self.proj_out(x) + x_in

        return ret
    
class Downsample(nn.Module):
    def __init__(self, channels):
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)
    

class Upsample(nn.Module):
    def __init__(self, channels):
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        bs, c, py, px = x.shape
        x = torch.reshape(x, (bs, c, py, 1, px, 1)).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)

        return self.conv(x)

# TimeStep Embedding
def timestep_emb(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    args = timesteps.numpy() * freqs
    embedding = torch.concat([torch.cos(args), torch.sin(args)])

    return torch.Tensor(embedding).reshape(1, -1)

class UNet(nn.Module):
    def __init__(self):
        self.time_emb = [
            nn.Linear(320, 1280),
            F.silu,
            nn.Linear(1280, 1280)
        ]

        # Input
        self.input_blocks = [
            [nn.Conv2d(4, 320, kernel_size=3, padding=1)],
            [ResBlock(320, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
            [ResBlock(320, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
            [Downsample(320)],
            [ResBlock(320, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
            [ResBlock(640, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
            [Downsample(640)],
            [ResBlock(640, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
            [ResBlock(1280, 1280, 1280),
             SpatialTransformer(1280, 768, 8, 160)],
            [Downsample(1280)],
            [ResBlock(1280, 1280, 1280)],
            [ResBlock(1280, 1280, 1280)]
        ]

        # Mid  
        self.middle_block = [
            ResBlock(1280, 1280, 1280),
            SpatialTransformer(1280, 768, 8, 160),
            ResBlock(1280, 1280, 1280)
        ]

        # Output
        self.output_blocks = [
            [ResBlock(2560, 1280, 1280)],
            [ResBlock(2560, 1280, 1280)],
            [ResBlock(2560, 1280, 1280), Upsample(1280)],
            [ResBlock(2560, 1280, 1280),
             SpatialTransformer(1280, 768, 8, 160)],
            [ResBlock(2560, 1280, 1280),
             SpatialTransformer(1280, 768, 8, 160)],
            [ResBlock(1920, 1280, 1280), SpatialTransformer(
                1280, 768, 8, 160), Upsample(1280)],
            [ResBlock(1920, 1280, 640), SpatialTransformer(
                640, 768, 8, 80)],  # 6
            [ResBlock(1280, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
            [ResBlock(960, 1280, 640), SpatialTransformer(
                640, 768, 8, 80), Upsample(640)],
            [ResBlock(960, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
            [ResBlock(640, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
            [ResBlock(640, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
        ]
        self.out = [
            nn.GroupNorm(320),
            F.silu,
            nn.Conv2d(320, 4, kernel_size=3, padding=1)
        ]

    # Try this !!!
    def sequential(self, x, layers):
        for l in layers:
            x = l(x)
        return x

    def forward(self, x, timesteps=None, context=None):
        t_emb = timestep_emb(timesteps, 320)
        emb = self.sequential(t_emb, self.time_emb)

        def run(x, bb):
            if isinstance(bb, ResBlock):
                bb(x, emb)
            elif isinstance(bb, SpatialTransformer):
                x = bb(x, context)
            else:
                x = bb(x)
            
            return x
        
        saved_inp = []
        for i, b in enumerate(self.input_blocks):
            for bb in b:
                x = run(x, bb)
            saved_inp.append(x)
        for bb in self.middle_block:
            x = run(x, bb)

        for i, b in enumerate(self.output_blocks):
            x = torch.cat(saved_inp.pop(), dim=1)
            for bb in b:
                x = run(x, bb)
        
        return self.sequential(x, self.out)
