from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # All Conv2d have stride = 1, AvgPool2d is performed after second Conv2d when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        
        self.conv3 = nn.Conv2d(inplanes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * self.expansion:
            # Downsampling Layer is prepended with AvgPool2d, and subsequent Conv2d has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))
        
    def forward(self, x: torch.Tensor):
        identity = x
        
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)

        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=2).permute(2, 0, 1) # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0) # (HW+1)NC
        x += self.pos_emb[:, None, :].to(x.dtype) # (HW+1)NC
        x = F.multi_head_attention_forward(
            query = x[:1], key = x, value = x,
            embed_dim_to_check = x.shape[-1],
            q_proj_weight = self.q_proj.weight, 
            k_proj_weight = self.k_proj.weight, 
            v_proj_weight = self.v_proj.weight,
            in_proj_weight  = None, 
            in_proj_bias = torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k = None, bias_v = None, add_zero_attn = False,
            dropout_p = 0, out_proj_weight = self.c_proj.weight, 
            out_proj_bias = self.c_proj.bias, use_separate_proj_weight = True,
            training = self.training, need_weights = True 
        )

        return x.squeeze(0)