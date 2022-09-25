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


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class ViT(nn.Module):
    def __init__(self, input_res: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_res = input_res
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_emb = nn.Parameter(scale * torch.randn(width))
        self.pos_emb = nn.Parameter(scale * torch.randn((input_res // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x) # [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [*, width, grid ** 2]
        x = x.permute(0, 2, 1) # [*, grid**2, width]
        x = torch.cat([self.class_emb.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.pos_emb.to(x.dtype)    
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x

class CLIP(nn.Module):
    def __init__(self, embed_dim: int, image_res: int, vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int, 
                 transformer_W: int, transformer_heads: int, transformer_layers: int):
        super().__init__()

        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers = vision_layers,
                output_dim = embed_dim,
                heads = vision_heads,
                input_resolution = image_res,
                width = vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = ViT(
                input_res = image_res,
                patch_size = vision_patch_size,
                width = vision_width,
                layers = vision_layers,
                heads = vision_heads,
                output_dim = embed_dim
            )
        
        self.transformer = Transformer(
            width = transformer_W,
            layers = transformer_layers,
            heads = transformer_heads,
            attn_mask = self.build_attn_mask()
        )

        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, transformer_W)
        self.pos_emb = nn.Parameter(torch.empty(self.context_length, transformer_W))
        self.ln_final = LayerNorm(transformer_W)
        
        self.text_proj = nn.Parameter(torch.empty(transformer_W, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_params()

    def initialize_params(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)
        
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std = attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std = proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std = fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std = proj_std)
            
        if self.text_proj is not None:
            nn.init.normal_(self.text_proj, std=self.transformer.width ** -0.5)

    def build_attn_mask(self):
        """
        Create Casual Attention Mask with full attention b/w visual tokens 
        """
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1) # zero out the lower diagonal

        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def encode_img(self, img):
        return self.visual(img.type(self.dtype))

    def encode_txt(self, txt):
        x = self.token_emb(txt).type(self.dtype) # [batch_size, n_ctx, d_model]
        x = x + self.pos_emb.type(self.dtype)
        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # [batch_size, n_ctx, transformer_width]
        # features from EOT embedding (eot_token is highest number in each sequence)
        x = x[torch.arange(x.shape[0]), txt.argmax(dim=-1)] @ self.text_proj

        return x 
    
    def forward(self, image, txt):
        img_features = self.encode_img(image)
        txt_features = self.encode_txt(txt)

        # Normalized Features
        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)

        # Cosine Similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_features @ txt_features.T()
        logits_per_txt = logits_per_image.T()

        return logits_per_image, logits_per_txt # [global_batch_size, global_batch_size]
    

def convert_weights(model: nn.Module):
    """ Convert model parameters to FP16 """
    def _convert_weights_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ['text_proj', 'proj']:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    
    model.apply(_convert_weights_fp16)

def build_model(state_dict: dict):
    vit = 'visual.proj' in state_dict

    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.pos_emb"].shape[0] - 1) ** 0.5)
        image_res = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.pos_emb"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.pos_emb"].shape[0]
        image_res = output_width * 32

    embed_dim = state_dict['text_proj'].shape[1]
    context_length = state_dict['pos_emb'].shape[0]
    vocab_size = state_dict['token_emb'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim = embed_dim,
        image_res = image_res, vision_layers = vision_layers, vision_width = vision_width, vision_patch_size = vision_patch_size,
        context_length = context_length, vocab_size = vocab_size, transformer_W = transformer_width, 
        transformer_heads = transformer_heads, transformer_layers = transformer_layers
    )

    for key in ['input_res', 'context_length', 'vocab_size']:
        if key in state_dict:
            del state_dict[key]
    
    convert_weights(model)
    model.load_state_dict(state_dict)

    return model.eval()