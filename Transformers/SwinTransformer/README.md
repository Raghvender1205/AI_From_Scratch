# Swin Transformer
It is a `Transformer` based model which uses `window-aware self attention`. In traditional transformer models, self attention mechanism calculates `attention weights` b/w all pairs of tokens in a sequence,. 

However, in the `window-aware` self attention, attention is only calculated b/w tokens that are within a fixed `window size`. This allows the model to better handle sequences of varying lengths and make use of `local context`. 
```
Local Context refers to the information relevant to a specific part of a sequence as opposed to the entire sequence. For example, in the sentence

    The cat sat on the mat

For the local context of "cat" may include "The", "sat" and local context of "mat" may include "on" and "the".
```

`Swin Transformer` also uses a method called `Swin-Block` which are blocks of `window aware` self attention and `point-wise` feed-forward layers which can then be stacked to make the model deeper.

## Implementation of Window Aware Self Attention
It is implemented by dividing the input sequence into a series of `overlapping windows`, and then calculating the self-attention weights within `each window` seperately. By this, the results are more efficient as the number of tokens in a window is much smaller than the total number of tokens in the sequence.

It also allows the model to better capture the dependencies b/w nearby tokens as the attention weights are only calculated b/w tokens that are within a fixed window size.

## Model Architecture

<img src="https://amaarora.github.io/images/swin-transformer.png"/>

### 1. Patch Partition/Embeddings

https://amaarora.github.io/2022/07/04/swintransformerv1.html#introduction