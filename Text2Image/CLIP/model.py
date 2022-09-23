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

class ModifiedResNet(nn.Module):
    """
    This ResNet has some modifications in the Architecture
    - There are 3 "stem" convolutions as opposed to 1, with an AvgPool2d instead of MaxPool2d
    - Performs anti-aliasing strided Conv2d, where AvgPool2d is prepended to Conv2d with stride > 1
    - Final Pooling Layer is a QKV Attention instead of AvgPool2d
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # 3 Layer Stem Convolutions
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(width // 2, width // 2,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # Residual Layers
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0]) 
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32 # ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [BottleNeck(self._inplanes, planes, stride)]

        self._inplanes = planes * BottleNeck.expansion
        for _ in range(1, blocks):
            layers.append(BottleNeck(self._inplanes, planes))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))

            return x

        x = x.dtype(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
    
        return x


class LayerNorm(nn.Module):
    """ A Subclass of nn.LayerNorm to handle FP16 """
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))

        return ret.type(orig_type)
    
class QuickGELU(nn.Module):
    """ Fast version of nn.GELU activation """
    def forward(self, x: torch.Tensor):
        x * F.sigmoid(1.702 * x)