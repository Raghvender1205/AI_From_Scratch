# https://arxiv.org/pdf/2112.10752.pdf
import os
import gzip
import math
import re
import traceback
from functools import lru_cache
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
    pass

class GeGELU(nn.Module):
    pass

class BasicTransformerBlock(nn.Module):
    pass