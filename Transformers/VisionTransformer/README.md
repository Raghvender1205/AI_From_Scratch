# ViT: An Image is Worth 16x16 Words
This is the overall architecture of a `Vision Transformer`

<img src="https://amaarora.github.io/images/ViT.png"/>
The functioning is as follows

1. Split images into patches
2. Get linear embeddings from each patch known as `Patch Embeddings`
3. Add `Position Embeddings` and `[cls]` token to each patch embeddings.
4. Pass through the `Transformer Encoder` and get the output values for each of the `[cls]` tokens.
5. Pass the representations of `[cls]` tokens through a `MLP Head` to get predictions.

Let's take an `224x224` image. We convert this image into patches of patch size `16x16`. We can have these patches in a straight line where the first patch comes from the `top-left` of the input image and last patch from the `bottom-right`. We get patch as `16 x 16 x 3`.

Now, we pass these patches through a `Linear Projection Layer` to get `1 x 768` vector representation of each image patch. These representations are known as `Patch Embeddings`. As we get `196` total patches so the total size of the `Patch Embedding Matrix` is `196 x 768`.

Now, we take this `Patch Embedding Matrix` and prepend a `[cls]` token to this sequence and then add `Positional Embeddings`. So, the size becomes `197 x 768`.

Then, we pass these preprocessed `Patch Embeddings` with Positional Information and prepended `[cls]` token to the `Transformer Encoder` and get learned representations of the `[cls]` token. We pass the output from the Transformer Encoder to the `MLP Head` to get the class predictions of size `1 x 768`.

## Patch Embeddings
To get `patch embeddings`, we split image into fixed-size patches and linearly embed each one of them using a `linear projection layer`. However, it is also possible to combine both steps using a `Conv2D` operation.

If we set `out_channels` to `768` and both `kernel_size` and `stride` to `16`, once we perform `Conv2D` operation we get `patch embeddings` of size `196 x 768` like this

```python
x = torch.randn(1, 3, 224, 224)
conv = nn.Conv2d(3, 768, 16, 16)
conv(x).reshape(-1, 196).transpose(0,1).shape

>> torch.Size([196, 768])
```

## [cls] Token and Position Embeddings

<img src="https://amaarora.github.io/images/vit-03.png"/>

A `[cls]` token is vector of `1 x 768`. We prepend it to the `Patch Embeddings` to get the updated size of `197 x 768`.

Then, we add `Postional Embeddings` of size `197 x 768` to the `Patch Embeddings` with `[cls]` token to get combined embeddings which can then be fed to the `Transformer Encoder`.

## Transformer Encoder
`Transformer Encoder` consists of `MultiHead Attention` and `MLP` blocks. `LayerNorm` is used before every block and residual connections after every block. A single is visualized as 

<img src="https://amaarora.github.io/images/vit-07.png"/>

First Layer accepts `combined embeddings` of shape `197 x 768` as input. There are 12 such layers in `ViT`. Inside the layer, the inputs are passed through `LayerNorm` and then fed to `MultiHead Attention` block.

Inside the `MultiHead Attention`, the inputs are converted to `197 x 2304 (768*3)` shape using `Linear` to get <b>`qkv`</b> matrix. Now, we reshape this `qkv` matrix into `197 x 3 x 768` where each of 3 matrices of shape `197 x 768` represent `q`, `k`, `v`. 

These matrices are then further reshaped into `12 x 197 x 64` to represent 12 attention heads. Now, we perform `Attention` inside the `MultiHead Attention Block` given by $Attention(qkv) = softmax(qk^T/\sqrt{d_x})*v$

Once we get the outputs from `Multi-Head Attention` block, these are added to the `skip connections` to get final outputs that again get passed to `LayerNorm` before being fed to `MLP` block.

`MLP` block consists of 2 `Linear` layers and a `GeLU` non-linearity. The outputs from `MLP` block are again added to the skip connections to get final output from 1 layer of the `Transformer Encoder`.

<img src="https://amaarora.github.io/images/vit-06.png"/>

Above is the Transformer Encoder. A single encoder contains 12 layers in which the outputs from the first layer are fed into the second layer and so on till 12th layer outputs are then fed to the `MLP` head to get class predictions.

## Implementation
ViT Implementation includes
1. Patch Embeddings
2. MLP 
3. Attention and Multi-Head Attention
4. Block
5. Main Architecture

### 1. Patch Embeddings
```python 

```

It can found at `vit.py`.