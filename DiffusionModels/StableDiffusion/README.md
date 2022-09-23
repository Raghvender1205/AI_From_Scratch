# Stable Diffusion
Stable Diffusion 

Paper: https://arxiv.org/pdf/2112.10752.pdf

Model Card: https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md

## References
1. https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py

2. https://colab.research.google.com/drive/1zVTa4mLeM_w44WaFwl7utTaa6JcaH1zK#scrollTo=eBnvJn3HKPjp

### TODO
1. Stable Diffusion for PyTorch
2. Improve Quality of images generated

### Updates
- Currently working on Stable Diffusion on PyTorch on this Error
```
Traceback (most recent call last):
  File "D:\AI_FROM_SCRATCH\DiffusionModels\StableDiffusion\stable_diffusion_torch.py", line 781, in <module>
    context = model.cond_stage_model.embedding(phrase)
  File "D:\Development\Python\Python3.10\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "D:\AI_FROM_SCRATCH\DiffusionModels\StableDiffusion\stable_diffusion_torch.py", line 582, in forward
    input_ids, pos_ids = x
ValueError: too many values to unpack (expected 2)
```

