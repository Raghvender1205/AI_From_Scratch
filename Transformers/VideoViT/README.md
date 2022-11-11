# Video Vision Transformer (ViViT)

Paper: https://arxiv.org/abs/2103.15691

## Tublet Embedding
In `ViT`, an image is divided into patches, which is then spatially flattened known as `tokenization`. `Uniform frame Sampling` is a tokenization scheme in which we sample `frames` from the video and perform `ViT Tokenization`.

<img src="https://i.imgur.com/aaPyLPX.png"/>

`Tublet Embedding` is different in terms of capturing `temporal information` from video. First, we extract <b>volumes</b> from video -- these volumes contain `patches` of the `frame` and the `temporal information` as well. The volumes are then flattened to build `video tokens`.

<img src="https://i.imgur.com/9G7QTfV.png"/>

## Variants of ViViT
There are 4 variants of `ViViT`

1. Spatio-temporal Attention
2. Factorized Encoder
3. Factorized Self-Attention
4. Factorized Dot-product Attention