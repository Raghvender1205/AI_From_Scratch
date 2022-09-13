# X-UNet
U-Net Model with `Efficient Attention`

## Usage
```python
import torch
from xunet import XUnet

unet = XUnet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    nested_unet_depths = (7, 4, 2, 1),     # nested unet depths, from unet-squared paper
    consolidate_upsample_fmaps = True,     # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
)

img = torch.randn(1, 3, 256, 256)
out = unet(img) # (1, 3, 256, 256)
```
For `3D (Video / CT Scans)`
```python
import torch
from x_unet import XUnet

unet = XUnet(
    dim = 64,
    frame_kernel_size = 3,                 # set this to greater than 1
    dim_mults = (1, 2, 4, 8),
    nested_unet_depths = (5, 4, 2, 1),     # nested unet depths, from unet-squared paper
    consolidate_upsample_fmaps = True,     # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
    weight_standardize = True
)

video = torch.randn(1, 3, 10, 128, 128)    # (batch, channels, frames, height, width)
out = unet(video) # (1, 3, 10, 128, 128)
```