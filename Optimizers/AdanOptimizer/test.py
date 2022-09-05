from adan import Adan
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(16, 16),
    nn.GELU()
)

optim = Adan(
    model.parameters(),
    lr = 1e-3,
    betas=(0.02, 0.08, 0.01), # beta-3 tuning is the most sensitive
    weight_decay=0.02 
)

# Train
for _ in range(10):
    loss = model(torch.randn(16)).sum()
    loss.backward()
    optim.step()
    optim.zero_grad()

    print(loss)

"""
$ python test.py 
tensor(2.1851, grad_fn=<SumBackward0>)
tensor(0.7503, grad_fn=<SumBackward0>)
tensor(3.2771, grad_fn=<SumBackward0>)
tensor(0.9880, grad_fn=<SumBackward0>)
tensor(0.8554, grad_fn=<SumBackward0>)
tensor(10.8852, grad_fn=<SumBackward0>)
tensor(2.6946, grad_fn=<SumBackward0>)
tensor(10.2375, grad_fn=<SumBackward0>)
tensor(-0.0653, grad_fn=<SumBackward0>)
tensor(3.2963, grad_fn=<SumBackward0>)
"""