from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Denoising Diffusion Probablistic Model
class DDPM(nn.Module):
    def __init__(self, eps_model: nn.Module, betas: Tuple[float, float],
                    n_T: int, criterion: nn.Module = nn.MSELoss()) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model
        
        # register_buffer allows to freely access tensors by name.
        for k, v in ddpm_schedules()


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    beta_t = (beta1 - beta2) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t"
    }