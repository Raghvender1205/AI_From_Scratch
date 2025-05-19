import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps) 

# Constants
device = 'cuda' if torch.cuda.is_available() else 'cpu'
T = 200 # Diffusion Steps
beta = linear_beta_schedule(T)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0).to(device) 


def q_sample(x0, t, noise=None):
    """
    Forward Diffusion Process
    """
    if noise is None:
        noise = torch.rand_like(x0)
    sqrt_alpha_bar = alpha_bar[t].sqrt().view(-1, 1).to(x0.device)
    sqrt_one_minus_bar = (1 - alpha_bar[t]).sqrt().view(-1, 1).to(x0.device)

    return sqrt_alpha_bar * x0 + sqrt_one_minus_bar * noise

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.model = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x, t: torch.Tensor):
        t = t.unsqueeze(-1).float() / T # normalize Time
        time_emb = self.time_mlp(t)
        x_input = torch.cat([x, time_emb], dim=1)

        return self.model(x_input)
    
@torch.no_grad()
def p_sample_loop(model, shape):
    """
    Reverse Process
    """
    x = torch.randn(shape).to(device)
    for t in reversed(range(T)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        mu = model(x, t_tensor)
        if t > 0:
            noise = torch.rand_like(x)
            x = mu + beta[t].sqrt().to(device) * noise
        else:
            x = mu
        
    return x
    
# Training
def train(model, dataloader, optimizer, n_epochs=5):
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for x, _ in pbar:
            x = x.view(x.size(0), -1).to(device)
            t = torch.randint(0, T, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            x_t = q_sample(x, t, noise=noise)

            pred_x0 = model(x_t, t)
            loss = F.mse_loss(pred_x0, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}: Avg Loss = {epoch_loss / len(dataloader):.4f}")


# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)])  # [-1, 1]
mnist = datasets.MNIST(root='data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

# Load model
model = SimpleMLP(input_dim=28*28).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
train(model, dataloader, optimizer, n_epochs=2)

# Sample
samples = p_sample_loop(model, (16, 28*28))
samples = samples.view(-1, 28, 28).cpu()
samples = (samples + 1) / 2  # convert from [-1, 1] to [0, 1]

# Plot
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i].numpy(), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()