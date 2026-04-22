"""
Diffusion Model for CHDP (Section 3.2)

Implements:
- Forward diffusion process (Eq. 1-2)
- Reverse diffusion sampling (Eq. 1, 5, 6)
- Noise schedule and sampling utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DiffusionSchedule:
    """
    Noise schedule for diffusion process.
    
    Implements variance schedule β_i and derived quantities ᾱ_i.
    """
    
    def __init__(
        self,
        num_steps: int = 15,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "linear"
    ):
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule_type == "cosine":
            s = 0.008
            steps = torch.arange(num_steps + 1) / num_steps
            alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = 1 - alphas
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def to(self, device: torch.device):
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self


class DiffusionProcess:
    """
    Forward and reverse diffusion process.
    
    Forward: q(a_i | a_0) = N(a_i; √ᾱ_i a_0, (1-ᾱ_i)I)  (Eq. 1)
    Reverse: p_θ(a_{i-1} | a_i, s) = N(a_{i-1}; μ_θ(a_i, s, i), σ_i²I)  (Eq. 5)
    """
    
    def __init__(self, schedule: DiffusionSchedule):
        self.schedule = schedule
    
    def forward_diffusion(
        self,
        a_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process (Eq. 1).
        
        Samples a_t ~ q(a_t | a_0) = N(a_t; √ᾱ_t a_0, (1-ᾱ_t)I)
        
        Args:
            a_0: Clean action tensor [batch, dim]
            t: Timestep tensor [batch]
            noise: Optional noise, if None samples new noise
            
        Returns:
            a_t: Noisy action at timestep t
            noise: The noise used
        """
        if noise is None:
            noise = torch.randn_like(a_0)
        
        sqrt_alphas_cumprod_t = self.schedule.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        a_t = sqrt_alphas_cumprod_t * a_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return a_t, noise
    
    def compute_loss(
        self,
        a_0: torch.Tensor,
        noise_predictor: nn.Module,
        state: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute diffusion loss (Eq. 2).
        
        L_d(θ) = E[||ε - ε_θ(√ᾱ_i a_0 + √(1-ᾱ_i)ε, s, i)||²]
        
        Args:
            a_0: Clean action [batch, dim]
            noise_predictor: ε_θ network
            state: State s for conditioning
            condition: Additional conditioning (e.g., codeword for continuous policy)
            
        Returns:
            loss: Scalar diffusion loss
        """
        batch_size = a_0.shape[0]
        device = a_0.device
        
        t = torch.randint(0, self.schedule.num_steps, (batch_size,), device=device)
        a_t, noise = self.forward_diffusion(a_0, t)
        
        if condition is not None:
            noise_pred = noise_predictor(a_t, state, t, condition)
        else:
            noise_pred = noise_predictor(a_t, state, t)
        
        loss = F.mse_loss(noise, noise_pred)
        return loss
    
    def reverse_diffusion_step(
        self,
        a_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
        state: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        model_variance: bool = False
    ) -> torch.Tensor:
        """
        Single reverse diffusion step (Eq. 5, 6).
        
        Samples a_{t-1} ~ p_θ(a_{t-1} | a_t, s)
        
        Args:
            a_t: Current noisy action [batch, dim]
            t: Current timestep [batch]
            noise_pred: Predicted noise ε_θ(a_t, s, t)
            state: State conditioning
            condition: Additional conditioning
            model_variance: Whether to use learned variance
            
        Returns:
            a_{t-1}: Denoised action
        """
        sqrt_alphas_cumprod_t = self.schedule.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        a_0_pred = (a_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t.clamp(min=1e-8)
        a_0_pred = a_0_pred.clamp(-1, 1)
        
        posterior_mean_coef1_t = self.schedule.posterior_mean_coef1[t].view(-1, 1)
        posterior_mean_coef2_t = self.schedule.posterior_mean_coef2[t].view(-1, 1)
        
        mean = (
            posterior_mean_coef1_t * a_0_pred +
            posterior_mean_coef2_t * a_t
        )
        
        if model_variance:
            posterior_variance_t = self.schedule.posterior_variance[t].view(-1, 1)
        else:
            posterior_variance_t = self.schedule.betas[t].view(-1, 1)
        
        if t[0] > 0:
            noise = torch.randn_like(a_t)
        else:
            noise = torch.zeros_like(a_t)
        
        a_prev = mean + torch.sqrt(posterior_variance_t) * noise
        return a_prev
    
    @torch.no_grad()
    def sample(
        self,
        noise_predictor: nn.Module,
        state: torch.Tensor,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Reverse diffusion sampling (Eq. 1).
        
        Generates action from noise: a_0 ~ p_θ(a_0 | s)
        
        Args:
            noise_predictor: ε_θ network
            state: State conditioning [batch, state_dim]
            shape: Shape of action tensor (batch, action_dim)
            condition: Additional conditioning (codeword for continuous)
            num_steps: Number of denoising steps (default: schedule.num_steps)
            
        Returns:
            a_0: Generated action
        """
        device = state.device
        batch_size = state.shape[0]
        
        if num_steps is None:
            num_steps = self.schedule.num_steps
        
        a_t = torch.randn(shape, device=device)
        
        for i in reversed(range(num_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            if condition is not None:
                noise_pred = noise_predictor(a_t, state, t, condition)
            else:
                noise_pred = noise_predictor(a_t, state, t)
            
            a_t = self.reverse_diffusion_step(
                a_t, t, noise_pred, state, condition
            )
        
        return a_t
